import argparse
import math

import torch

from semg_jepa.architecture import BaselineCTCModel
from semg_jepa.cached_dataset import CachedRawEMGDataset
from semg_jepa.ctc_utils import evaluate, grid_search
from semg_jepa.tokenizers import build_tokenizer

GRID_BEAM_WIDTHS = [50, 100, 200, 300]
GRID_ALPHAS = [0.5, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5]
GRID_BETAS = [0.0, 0.5, 1.0, 1.5, 1.85, 2.0, 2.5]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="/scratch/cr4206/sEMG-unpaired-text/data")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="One or more checkpoint paths. Each is evaluated independently.")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--method", choices=["greedy", "beam"], default="beam")
    p.add_argument("--beam-width", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.90)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--lm-path", default="/scratch/cr4206/sEMG-unpaired-text/data/lm.binary")
    p.add_argument("--unigrams-path", default="/scratch/cr4206/sEMG-unpaired-text/data/unigrams.txt")
    p.add_argument("--grid-search", action="store_true",
                   help="Tune (beam_width, alpha, beta) on dev, then evaluate --split with the best.")
    # Must match how the checkpoint was trained.
    p.add_argument("--unit", choices=["char", "subword", "phoneme"], default="char")
    p.add_argument("--subword-model", default=None)
    p.add_argument("--phoneme-dict", default=None)
    p.add_argument("--conv-strides", type=int, nargs="+", default=[2, 2, 2])
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    conv_strides = tuple(args.conv_strides)
    factor = math.prod(conv_strides)
    tokenizer = build_tokenizer(args.unit, subword_model=args.subword_model,
                                phoneme_dict=args.phoneme_dict)

    eval_dataset = CachedRawEMGDataset(args.cache_dir, args.split,
                                       tokenizer=tokenizer, downsample_factor=factor)
    dev_dataset = (CachedRawEMGDataset(args.cache_dir, "dev",
                                       tokenizer=tokenizer, downsample_factor=factor)
                   if args.grid_search and args.split != "dev" else eval_dataset)

    for ckpt in args.checkpoints:
        print(f"\n=== Checkpoint: {ckpt} ===", flush=True)
        model = BaselineCTCModel(vocab_size=tokenizer.vocab_size, conv_strides=conv_strides).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)

        if args.grid_search:
            results = grid_search(model, dev_dataset, device,
                                  GRID_BEAM_WIDTHS, GRID_ALPHAS, GRID_BETAS,
                                  lm_path=args.lm_path, unigrams_path=args.unigrams_path, 
                                  batch_size=1)
            print(f"  grid search on dev (top 10 of {len(results)}):")
            for bw, a, b, wer, cer in results[:10]:
                print(f"    bw={bw:>3} alpha={a:.2f} beta={b:.2f}  WER={wer:.4f} CER={cer:.4f}")
            best_bw, best_a, best_b, _, _ = results[0]
            print(f"  best on dev: bw={best_bw} alpha={best_a} beta={best_b}", flush=True)
            wer, cer = evaluate(model, eval_dataset, device, method="beam",
                                beam_width=best_bw, alpha=best_a, beta=best_b,
                                lm_path=args.lm_path, unigrams_path=args.unigrams_path)
        else:
            wer, cer = evaluate(model, eval_dataset, device, method=args.method,
                                beam_width=args.beam_width, alpha=args.alpha, beta=args.beta,
                                lm_path=args.lm_path, unigrams_path=args.unigrams_path, 
                                batch_size=1)
        print(f"  WER ({args.split}): {wer:.4f}   CER ({args.split}): {cer:.4f}", flush=True)


if __name__ == "__main__":
    main(parse_args())
