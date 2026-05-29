"""Build pyctcdecode unigrams from a generic English vocabulary.

Why this exists
---------------
pyctcdecode (used as a ctcdecode replacement) takes a `unigrams=[...]` list to
guide its beam search. Without it, every character n-gram is a candidate word
during decoding and quality degrades.

Why NOT extract unigrams from train/dev/test transcripts
--------------------------------------------------------
A vocabulary derived from the dataset's own transcripts artificially boosts
in-domain WER: the decoder cannot output any word the dataset never saw, even
though at deployment the model is supposed to handle general English. That is
data leakage and we explicitly do not want it.

What this module does
---------------------
1. Pulls the LibriSpeech vocabulary (200k words, public, independent of our
   silent-speech dataset) from openslr.org.
2. Normalizes each word with the same pipeline `TextTransform.clean_text`
   uses on the targets (unidecode + lowercase + drop everything outside
   [a-z]). Tokens that collapse to nothing are dropped.
3. Filters through KenLM: keeps only words the LM itself knows (non-OOV).
   Removes pyctcdecode's "this word isn't in the LM" warning surface.
4. Writes the result to disk.

Re-run after replacing the LM file or to refresh the wordlist.
"""

import re
import time
import urllib.request
from pathlib import Path

LIBRISPEECH_VOCAB_URL = "https://www.openslr.org/resources/11/librispeech-vocab.txt"


def build_unigrams(lm_path, out_path, vocab_source=LIBRISPEECH_VOCAB_URL):
    """Build a leakage-free unigrams file from a generic English vocabulary.

    Returns the list of unigrams written.
    """
    if isinstance(vocab_source, str) and vocab_source.startswith(("http://", "https://")):
        print(f"[unigrams] fetching {vocab_source}", flush=True)
        with urllib.request.urlopen(vocab_source) as r:
            raw = r.read().decode().splitlines()
    else:
        raw = Path(vocab_source).read_text().splitlines()
    raw = [w.strip() for w in raw if w.strip()]
    print(f"[unigrams] raw vocab: {len(raw)}", flush=True)

    from unidecode import unidecode
    alpha_re = re.compile(r"[^a-z]")
    candidates = set()
    for w in raw:
        w = re.sub(alpha_re, "", unidecode(w).lower())
        if w:
            candidates.add(w)
    print(f"[unigrams] after clean+filter to [a-z]+: {len(candidates)}", flush=True)

    import kenlm
    m = kenlm.Model(str(lm_path))
    t0 = time.perf_counter()
    known = set()
    for w in candidates:
        score = next(iter(m.full_scores(w, bos=True, eos=False)))
        if not score[2]:  # not OOV
            known.add(w)
    print(f"[unigrams] after LM-known filter: {len(known)}  ({time.perf_counter()-t0:.1f}s)", flush=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_known = sorted(known)
    out_path.write_text("\n".join(sorted_known) + "\n")
    print(f"[unigrams] wrote {out_path}", flush=True)
    return sorted_known


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--lm-path", default="/scratch/cr4206/sEMGencoderJEPA/data/lm.binary")
    p.add_argument("--out", default="/scratch/cr4206/sEMGencoderJEPA/data/unigrams.txt")
    p.add_argument("--vocab-source", default=LIBRISPEECH_VOCAB_URL)
    args = p.parse_args()
    build_unigrams(args.lm_path, args.out, args.vocab_source)
