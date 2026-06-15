"""Build an OPEN-VOCAB phoneme lexicon from a wordlist (default: LibriSpeech unigrams).

Why: the dataset phoneme dict (`data/tokenizers/phoneme_g2p.dict`) is g2p'd from
train+dev+test transcripts, so decoding phones -> words through it is conditioned on the
closed dataset vocabulary (it can "see" the test words). To decode open-vocab — and to
reuse the *same* KenLM word LM that backs the char/subword beams — we build a lexicon from
`data/unigrams.txt` (the ~180k leakage-free LibriSpeech vocab the char beam already uses).

Runs `g2p_en` once over the wordlist and writes three index-aligned artifacts:
  1. <out_dict>      CMUdict format ("WORD  P H O N E S")  — loadable by PhonemeTokenizer;
  2. <out_lexicon>   torchaudio ctc_decoder lexicon ("word ph1 ph2 ... |")  — words
                     lowercased, spelling ends with the word-separator token `|`;
  3. <out_tokens>    the 49 CTC token strings, index-aligned to the model logits
                     [ARPABET(47), '|'(47), <blank>(48)] — for torchaudio `tokens=`.

Usage (CPU, ~once):
    PYTHONPATH=/scratch/cr4206/sEMG-unpaired-text \
    python scripts/build_lexicon_from_unigrams.py
"""
from __future__ import annotations

import argparse
import os
import time

from semg_jepa.tokenizers import ARPABET, PhonemeTokenizer, clean_text

BLANK_TOKEN = "<blank>"


def write_tokens_file(path):
    """49 tokens, index-aligned to the CTC logits: phones, '|', blank (last)."""
    symbols = list(ARPABET) + [PhonemeTokenizer.WORD_SEP]  # 48 emit classes (0..47)
    tokens = symbols + [BLANK_TOKEN]                       # blank at index 48
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(tokens) + "\n")
    print(f"[build_lexicon] wrote {len(tokens)} tokens -> {path}", flush=True)
    return tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wordlist", default="data/unigrams.txt",
                   help="one lowercase word per line (default: LibriSpeech unigrams)")
    p.add_argument("--out-dict", default="data/tokenizers/phoneme_librispeech.dict")
    p.add_argument("--out-lexicon", default="data/tokenizers/lexicon_librispeech.txt")
    p.add_argument("--out-tokens", default="data/tokenizers/phoneme_tokens.txt")
    p.add_argument("--max-words", type=int, default=0, help="0 = all (debug subset otherwise)")
    args = p.parse_args()

    write_tokens_file(args.out_tokens)

    with open(args.wordlist, "r", encoding="utf-8") as f:
        words = []
        seen = set()
        for line in f:
            w = clean_text(line.strip())
            w = w.split()[0] if w.split() else ""
            if w and w not in seen:
                seen.add(w)
                words.append(w)
    if args.max_words:
        words = words[: args.max_words]
    print(f"[build_lexicon] {len(words)} unique words from {args.wordlist}", flush=True)

    from g2p_en import G2p  # neural G2P; CMUdict lookup fast, OOV via model

    g2p = G2p()
    arp = set(ARPABET)
    os.makedirs(os.path.dirname(args.out_dict), exist_ok=True)
    t0 = time.time()
    n = 0
    with open(args.out_dict, "w") as fd, open(args.out_lexicon, "w") as fl:
        for i, w in enumerate(words):
            phones = [ph.strip().rstrip("012").lower() for ph in g2p(w)]
            phones = [ph for ph in phones if ph in arp]
            if not phones:
                continue
            fd.write(w.upper() + "  " + " ".join(phones) + "\n")
            fl.write(w + " " + " ".join(phones) + " " + PhonemeTokenizer.WORD_SEP + "\n")
            n += 1
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(words)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[build_lexicon] wrote {n} entries -> {args.out_dict} and {args.out_lexicon} "
          f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
