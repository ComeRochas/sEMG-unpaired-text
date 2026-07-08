"""Pluggable target-unit tokenizers for CTC silent-speech training.

Phase-2 axis: predict **characters**, **subwords**, **phonemes**, or **HuBERT units**
from EMG, and re-train a supervised baseline per unit. A tokenizer is the single source of truth
for the CTC vocabulary and the decode/score rendering, so the rest of the pipeline
(dataset → model vocab → CTC blank → beam decoder → WER/CER) stays unit-agnostic.

Each tokenizer exposes:
- ``vocab_size``      : number of emit classes (CTC blank is appended at ``blank_index``).
- ``blank_index``     : == ``vocab_size`` (logits layout is ``[..vocab.., blank]``).
- ``labels``          : ``list[str]`` of length ``vocab_size`` for ``pyctcdecode``.
- ``supports_word_lm``: whether the KenLM *word* LM + unigrams apply to ``labels``.
- ``clean_text(s)``   : text normalization (ASCII, strip punctuation, lowercase).
- ``text_to_int(s)``  : encode a transcript to target ids.
- ``int_to_text(ids)``: render a hypothesis id-sequence back to a scorable string.
- ``reference_text(s)``: render the gold transcript at this unit's granularity, so
  the WER/CER comparison is like-for-like (words for char/subword, phones for phoneme).

Build one with :func:`build_tokenizer`.
"""
from __future__ import annotations

import string
from pathlib import Path

import jiwer
from unidecode import unidecode

# Shared text normalization (identical to the phase-1 char `TextTransform`).
_CLEAN = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])


def clean_text(text: str) -> str:
    return _CLEAN(unidecode(text))


# ARPAbet inventory (mirrors `semg_jepa.data_utils.phoneme_inventory`, kept local so
# this module stays import-light — no librosa/soundfile pulled in just to tokenize).
ARPABET = [
    "aa", "ae", "ah", "ao", "aw", "ax", "axr", "ay", "b", "ch", "d", "dh", "dx",
    "eh", "el", "em", "en", "er", "ey", "f", "g", "hh", "hv", "ih", "iy", "jh", "k",
    "l", "m", "n", "nx", "ng", "ow", "oy", "p", "r", "s", "sh", "t", "th", "uh", "uw",
    "v", "w", "y", "z", "zh",
]


class BaseTokenizer:
    unit = "base"
    supports_word_lm = False
    # When True, the CTC targets are NOT a function of the transcript and must be
    # supplied to the dataset out-of-band (e.g. HuBERT units extracted from audio).
    # ``text_to_int``/``reference_text`` then have no meaning; the dataset/eval read
    # the gold id-sequence from a precomputed cache instead.
    targets_from_audio = False

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    def blank_index(self) -> int:
        return self.vocab_size

    @property
    def labels(self) -> list[str]:
        raise NotImplementedError

    def clean_text(self, text: str) -> str:
        return clean_text(text)

    def text_to_int(self, text: str) -> list[int]:
        raise NotImplementedError

    def int_to_text(self, ints) -> str:
        raise NotImplementedError

    def reference_text(self, text: str) -> str:
        """Gold rendering at this unit's granularity, for WER/CER scoring."""
        return self.clean_text(text)


class CharTokenizer(BaseTokenizer):
    """37-symbol character CTC vocab: ``a-z 0-9 space``. Phase-1 default."""

    unit = "char"
    supports_word_lm = True

    def __init__(self):
        self.chars = string.ascii_lowercase + string.digits + " "

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @property
    def labels(self) -> list[str]:
        return list(self.chars)

    def text_to_int(self, text: str) -> list[int]:
        return [self.chars.index(c) for c in self.clean_text(text)]

    def int_to_text(self, ints) -> str:
        return "".join(self.chars[i] for i in ints)


class SubwordTokenizer(BaseTokenizer):
    """SentencePiece subword CTC vocab. Train a model with ``scripts/train_subword.py``.

    SentencePiece marks word starts with ``▁``; ``pyctcdecode`` understands that
    marker as a BPE word boundary, so the KenLM *word* LM still applies.
    """

    unit = "subword"
    supports_word_lm = True

    def __init__(self, model_path: str):
        import sentencepiece as spm  # lazy: only needed for the subword unit

        model_path = str(model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"SentencePiece model not found: {model_path}. "
                "Train one with `python scripts/train_subword.py --vocab-size N`."
            )
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def labels(self) -> list[str]:
        return [self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())]

    def text_to_int(self, text: str) -> list[int]:
        return self.sp.encode(self.clean_text(text), out_type=int)

    def int_to_text(self, ints) -> str:
        # SentencePiece decode turns ``▁`` back into spaces → a word string.
        return self.sp.decode(list(ints))

    # reference_text inherited == clean_text (word string): we score subword
    # hypotheses at the word level after `sp.decode`.


class PhonemeTokenizer(BaseTokenizer):
    """ARPAbet phoneme CTC vocab + a word-separator token ``|``.

    Needs a grapheme-to-phoneme backend to encode transcripts. Two options,
    tried in order: a ``g2p_en.G2p`` instance (``pip install g2p_en``), or a
    CMUdict-style text file (``--phoneme-dict``) of ``WORD  P H O N E S`` lines.
    Word boundaries are encoded as ``|`` so the reported error rate is a
    phone-level error rate with word structure preserved.
    """

    unit = "phoneme"
    supports_word_lm = False  # no phoneme LM yet → beam falls back to no-LM

    WORD_SEP = "|"

    def __init__(self, g2p=None, dict_path: str | None = None):
        self.symbols = list(ARPABET) + [self.WORD_SEP]
        self._sym2id = {s: i for i, s in enumerate(self.symbols)}
        self._g2p = None
        self._dict = None
        if g2p is not None:
            self._g2p = g2p
        elif dict_path is not None:
            self._dict = self._load_dict(dict_path)
        else:
            try:
                from g2p_en import G2p  # type: ignore
                self._g2p = G2p()
            except Exception as e:
                raise RuntimeError(
                    "PhonemeTokenizer needs a G2P backend. Either `pip install g2p_en` "
                    "or pass a CMUdict file via --phoneme-dict. "
                    f"(g2p_en import failed: {e})"
                )

    @staticmethod
    def _load_dict(dict_path: str) -> dict[str, list[str]]:
        d: dict[str, list[str]] = {}
        with open(dict_path, "r", encoding="latin-1") as f:
            for line in f:
                if line.startswith(";;;") or not line.strip():
                    continue
                word, *phones = line.split()
                word = word.split("(")[0].lower()  # drop CMUdict (1)/(2) variants
                # nltk's cmudict data file inserts a variant index as a 2nd column
                # (e.g. "A 1 AH0"); skip a leading all-digit token so both the
                # classic "WORD  PH ..." and nltk "WORD N PH ..." formats parse.
                if phones and phones[0].isdigit():
                    phones = phones[1:]
                phones = [p.rstrip("012").lower() for p in phones]
                d.setdefault(word, phones)
        return d

    def _word_to_phones(self, word: str) -> list[str]:
        if self._dict is not None:
            return self._dict.get(word, [])
        # g2p_en path: returns phones (with stress digits) + spaces.
        phones = []
        for p in self._g2p(word):
            p = p.strip().rstrip("012").lower()
            if p in self._sym2id:
                phones.append(p)
        return phones

    def _encode_symbols(self, text: str) -> list[str]:
        words = self.clean_text(text).split()
        out: list[str] = []
        for i, w in enumerate(words):
            if i > 0:
                out.append(self.WORD_SEP)
            out.extend(self._word_to_phones(w))
        return out

    @property
    def vocab_size(self) -> int:
        return len(self.symbols)

    @property
    def labels(self) -> list[str]:
        return list(self.symbols)

    def text_to_int(self, text: str) -> list[int]:
        return [self._sym2id[s] for s in self._encode_symbols(text)]

    def int_to_text(self, ints) -> str:
        return " ".join(self.symbols[i] for i in ints)

    def reference_text(self, text: str) -> str:
        # Phone sequence (word boundaries as `|`); WER over this == phone error rate.
        return " ".join(self._encode_symbols(text))


class HubertUnitTokenizer(BaseTokenizer):
    """Discrete HuBERT acoustic units (k-means cluster IDs) as the CTC target.

    Unlike char/subword/phoneme, a unit sequence is a function of the **audio**, not
    the transcript, so it cannot be derived from ``text``. Targets are precomputed
    (``scripts/precompute_hubert_units.py``) from the parallel voiced audio and fed to
    the dataset via ``unit_targets`` (keyed by ``sample_id``); ``targets_from_audio``
    flags that to the dataset and the evaluator.

    ``vocab_size`` = k (the k-means K, 100 for the HuBERTVoc km100 model). There is no
    word LM, so beam decoding is disabled and the dev metric is a **unit error rate**
    (jiwer over the space-joined unit IDs). The real word-WER comes from re-synthesis
    (units -> vocoder -> ASR) in ``scripts/eval_hubert_resynth.py``.
    """

    unit = "hubert"
    supports_word_lm = False
    targets_from_audio = True

    def __init__(self, k: int = 100):
        self.k = int(k)

    @property
    def vocab_size(self) -> int:
        return self.k

    @property
    def labels(self) -> list[str]:
        return [str(i) for i in range(self.k)]

    def int_to_text(self, ints) -> str:
        return " ".join(str(int(i)) for i in ints)

    def text_to_int(self, text: str) -> list[int]:
        raise NotImplementedError(
            "HuBERT unit targets come from audio, not text. Precompute them with "
            "scripts/precompute_hubert_units.py and pass unit_targets to the dataset."
        )

    def reference_text(self, text: str) -> str:
        raise NotImplementedError(
            "HuBERT references are gold unit sequences (not text). The evaluator "
            "renders them from the cached unit ids via int_to_text."
        )


def build_tokenizer(unit: str = "char", *, subword_model: str | None = None,
                    phoneme_dict: str | None = None, phoneme_g2p=None,
                    hubert_k: int = 100) -> BaseTokenizer:
    """Factory selecting the target unit from a config string."""
    unit = (unit or "char").lower()
    if unit == "char":
        return CharTokenizer()
    if unit == "subword":
        if not subword_model:
            raise ValueError("unit='subword' requires --subword-model / subword_model.")
        return SubwordTokenizer(subword_model)
    if unit == "phoneme":
        return PhonemeTokenizer(g2p=phoneme_g2p, dict_path=phoneme_dict)
    if unit == "hubert":
        return HubertUnitTokenizer(k=hubert_k)
    raise ValueError(f"unknown unit: {unit!r} (expected char|subword|phoneme|hubert)")
