import string

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
from textgrids import TextGrid
from unidecode import unidecode

phoneme_inventory = [
    "aa", "ae", "ah", "ao", "aw", "ax", "axr", "ay", "b", "ch", "d", "dh", "dx",
    "eh", "el", "em", "en", "er", "ey", "f", "g", "hh", "hv", "ih", "iy", "jh", "k",
    "l", "m", "n", "nx", "ng", "ow", "oy", "p", "r", "s", "sh", "t", "th", "uh", "uw",
    "v", "w", "y", "z", "zh", "sil",
]


def dynamic_range_compression_torch(x: torch.Tensor, c: float = 1.0, clip_val: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * c)


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    return dynamic_range_compression_torch(magnitudes)


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    global mel_basis, hann_window
    key = f"{fmax}_{y.device}"
    if key not in mel_basis:
        mel = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect")
    y = y.squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(mel_basis[key], spec)
    return spectral_normalize_torch(spec)


def load_audio(filename, start=None, end=None, max_frames=None):
    audio, sample_rate = sf.read(filename)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if start is not None or end is not None:
        audio = audio[start:end]
    if sample_rate == 16000:
        audio = librosa.resample(audio, orig_sr=16000, target_sr=22050)
    else:
        assert sample_rate == 22050
    audio = np.clip(audio, -1, 1)
    mels = mel_spectrogram(
        torch.tensor(audio, dtype=torch.float32).unsqueeze(0),
        1024,
        80,
        22050,
        256,
        1024,
        0,
        8000,
        center=False,
    )
    mspec = mels.squeeze(0).T.numpy()
    if max_frames is not None and mspec.shape[0] > max_frames:
        mspec = mspec[:max_frames, :]
    return mspec


def double_average(x):
    f = np.ones(9) / 9.0
    return np.convolve(np.convolve(x, f, mode="same"), f, mode="same")


def get_emg_features(emg_data):
    xs = emg_data - emg_data.mean(axis=0, keepdims=True)
    frame_features = []
    for i in range(emg_data.shape[1]):
        x = xs[:, i]
        w = double_average(x)
        p = x - w
        r = np.abs(p)
        w_h = librosa.util.frame(w, frame_length=16, hop_length=6).mean(axis=0)
        p_w = np.squeeze(librosa.feature.rms(y=w, frame_length=16, hop_length=6, center=False), 0)
        p_r = np.squeeze(librosa.feature.rms(y=r, frame_length=16, hop_length=6, center=False), 0)
        z_p = np.squeeze(librosa.feature.zero_crossing_rate(p, frame_length=16, hop_length=6, center=False), 0)
        r_h = librosa.util.frame(r, frame_length=16, hop_length=6).mean(axis=0)
        s = abs(librosa.stft(np.ascontiguousarray(x), n_fft=16, hop_length=6, center=False))
        frame_features.append(np.stack([w_h, p_w, p_r, z_p, r_h], axis=1))
        frame_features.append(s.T)
    return np.concatenate(frame_features, axis=1).astype(np.float32)


class FeatureNormalizer:
    def __init__(self, feature_samples, share_scale=False):
        feature_samples = np.concatenate(feature_samples, axis=0)
        self.feature_means = feature_samples.mean(axis=0, keepdims=True)
        self.feature_stddevs = feature_samples.std() if share_scale else feature_samples.std(axis=0, keepdims=True)

    def normalize(self, sample):
        return (sample - self.feature_means) / self.feature_stddevs


def combine_fixed_length(tensor_list, length):
    total_length = sum(t.size(0) for t in tensor_list)
    if total_length % length != 0:
        pad_length = length - (total_length % length)
        tensor_list = list(tensor_list)
        tensor_list.append(torch.zeros(pad_length, *tensor_list[0].size()[1:], dtype=tensor_list[0].dtype, device=tensor_list[0].device))
        total_length += pad_length
    tensor = torch.cat(tensor_list, 0)
    return tensor.view(total_length // length, length, *tensor.size()[1:])


def decollate_tensor(tensor, lengths):
    b, s, d = tensor.size()
    tensor = tensor.view(b * s, d)
    results, idx = [], 0
    for length in lengths:
        results.append(tensor[idx:idx + length])
        idx += length
    return results


def read_phonemes(textgrid_fname, max_len=None):
    tg = TextGrid(textgrid_fname)
    phone_ids = np.zeros(int(tg["phones"][-1].xmax * 86.133) + 1, dtype=np.int64)
    phone_ids[:] = -1
    phone_ids[-1] = phoneme_inventory.index("sil")
    for interval in tg["phones"]:
        phone = interval.text.lower()
        if phone in ["", "sp", "spn"]:
            phone = "sil"
        if phone[-1] in string.digits:
            phone = phone[:-1]
        ph_id = phoneme_inventory.index(phone)
        phone_ids[int(interval.xmin * 86.133):int(interval.xmax * 86.133)] = ph_id
    if max_len is not None:
        phone_ids = phone_ids[:max_len]
    return phone_ids


class TextTransform:
    def __init__(self):
        self.transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
        self.chars = string.ascii_lowercase + string.digits + " "

    def clean_text(self, text):
        return self.transformation(unidecode(text))

    def text_to_int(self, text):
        return [self.chars.index(c) for c in self.clean_text(text)]

    def int_to_text(self, ints):
        return "".join(self.chars[i] for i in ints)
