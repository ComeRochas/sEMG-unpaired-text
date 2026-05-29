import json
import logging
import os
import random
import re
import string
from copy import copy

import numpy as np
import scipy
import torch

from .data_utils import TextTransform


def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, "highpass", fs=fs)
    return scipy.signal.filtfilt(b, a, signal)


def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)


def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1, 8):
        signal = notch(signal, freq * harmonic, sample_frequency)
    return signal


def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal)) / old_freq
    sample_times = np.arange(0, times[-1], 1 / new_freq)
    return np.interp(sample_times, times, signal)


def apply_to_all(function, signal_array, *args, **kwargs):
    return np.stack([function(signal_array[:, i], *args, **kwargs) for i in range(signal_array.shape[1])], 1)


def load_utterance(base_dir, index, remove_channels=None):
    """Load and preprocess a single utterance. Returns a dict with raw_emg,
    text, book_location, and ctc_length (number of 86 Hz frames)."""
    remove_channels = remove_channels or []
    index = int(index)
    raw_emg = np.load(os.path.join(base_dir, f"{index}_emg.npy"))
    before = os.path.join(base_dir, f"{index - 1}_emg.npy")
    after = os.path.join(base_dir, f"{index + 1}_emg.npy")
    raw_emg_before = np.load(before) if os.path.exists(before) else np.zeros([0, raw_emg.shape[1]])
    raw_emg_after = np.load(after) if os.path.exists(after) else np.zeros([0, raw_emg.shape[1]])

    x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], 0)
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    x = x[raw_emg_before.shape[0]:x.shape[0] - raw_emg_after.shape[0], :]
    emg_orig = apply_to_all(subsample, x, 689.06, 1000)

    for c in remove_channels:
        emg_orig[:, int(c)] = 0

    # T frames at ~86.13 Hz (689.06 / 8), offset of 8 matches Gaddy et al.
    T = (len(emg_orig) - 8) // 8
    raw = emg_orig[8:8 + 8 * T, :].astype(np.float32)
    raw = raw / 20
    raw = 50 * np.tanh(raw / 50.0)

    with open(os.path.join(base_dir, f"{index}_info.json")) as f:
        info = json.load(f)

    return {
        "raw_emg": raw,
        "text": info["text"],
        "book_location": (info["book"], info["sentence_index"]),
        "ctc_length": T,
    }


class EMGDirectory:
    def __init__(self, session_index, directory, silent, exclude_from_testset=False):
        self.session_index = session_index
        self.directory = directory
        self.silent = silent
        self.exclude_from_testset = exclude_from_testset

    def __lt__(self, other):
        return self.session_index < other.session_index


class SizeAwareSampler(torch.utils.data.Sampler):
    def __init__(self, emg_dataset, max_len):
        self.dataset = emg_dataset
        self.max_len = max_len

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        batch, batch_length = [], 0
        for idx in indices:
            raw_len, has_text = self.dataset._info_cache[idx]
            if not has_text:
                continue
            if raw_len > self.max_len:
                logging.warning("Example %s too long for configured max_len=%s", idx, self.max_len)
            if raw_len + batch_length > self.max_len:
                yield batch
                batch, batch_length = [], 0
            batch.append(idx)
            batch_length += raw_len


class EMGDataset(torch.utils.data.Dataset):
    def __init__(self, config, dev=False, test=False, no_testset=False):
        self.config = config

        if no_testset:
            devset, testset = [], []
        else:
            with open(config["testset_file"]) as f:
                testset_json = json.load(f)
            devset, testset = testset_json["dev"], testset_json["test"]

        directories = []
        for sd in config.get("silent_data_directories", []):
            for session_dir in sorted(os.listdir(sd)):
                directories.append(EMGDirectory(len(directories), os.path.join(sd, session_dir), True))
        has_silent = len(config.get("silent_data_directories", [])) > 0
        for vd in config.get("voiced_data_directories", []):
            for session_dir in sorted(os.listdir(vd)):
                directories.append(EMGDirectory(len(directories), os.path.join(vd, session_dir), False, exclude_from_testset=has_silent))

        raw_examples = []
        for directory_info in directories:
            for fname in os.listdir(directory_info.directory):
                m = re.match(r"(\d+)_info.json", fname)
                if m is None:
                    continue
                idx = int(m.group(1))
                with open(os.path.join(directory_info.directory, fname)) as f:
                    info = json.load(f)
                if info["sentence_index"] < 0:
                    continue
                location = [info["book"], info["sentence_index"]]
                in_test, in_dev = location in testset, location in devset
                if (test and in_test and not directory_info.exclude_from_testset) or \
                   (dev and in_dev and not directory_info.exclude_from_testset) or \
                   (not test and not dev and not in_test and not in_dev):
                    raw_examples.append((directory_info, idx, info))

        raw_examples.sort(key=lambda x: (x[0].session_index, x[1]))
        random.seed(0)
        random.shuffle(raw_examples)

        self.example_indices = [(d, i) for d, i, _ in raw_examples]
        # Cache (raw_1000hz_len, has_text) to avoid re-reading files in SizeAwareSampler.
        self._info_cache = [
            (sum(chunk[0] for chunk in info["chunks"]),
             any(c in string.ascii_letters for c in info["text"]))
            for _, _, info in raw_examples
        ]

        self.text_transform = TextTransform()

    def __len__(self):
        return len(self.example_indices)

    def subset(self, fraction):
        result = copy(self)
        n = int(fraction * len(self.example_indices))
        result.example_indices = self.example_indices[:n]
        result._info_cache = self._info_cache[:n]
        return result

    def __getitem__(self, i):
        directory_info, idx = self.example_indices[i]
        utt = load_utterance(
            directory_info.directory, idx,
            remove_channels=self.config.get("remove_channels", []),
        )
        T = utt["ctc_length"]
        text_int = np.array(self.text_transform.text_to_int(utt["text"]), dtype=np.int64)
        return {
            "raw_emg": torch.from_numpy(utt["raw_emg"]),
            "text": utt["text"],
            "text_int": torch.from_numpy(text_int),
            "session_ids": torch.full((T,), directory_info.session_index, dtype=torch.long),
            "silent": directory_info.silent,
            "book_location": utt["book_location"],
            "length": T,
        }

    @staticmethod
    def collate_raw(batch):
        return {
            "raw_emg": [ex["raw_emg"] for ex in batch],
            "lengths": [ex["length"] for ex in batch],
            "session_ids": [ex["session_ids"] for ex in batch],
            "silent": [ex["silent"] for ex in batch],
            "text_int": [ex["text_int"] for ex in batch],
            "text_int_lengths": [ex["text_int"].shape[0] for ex in batch],
            "text": [ex["text"] for ex in batch],
            "book_location": [ex["book_location"] for ex in batch],
        }
