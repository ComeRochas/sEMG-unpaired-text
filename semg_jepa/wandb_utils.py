from __future__ import annotations

import os
from datetime import datetime


def init_wandb(args, default_name_prefix: str | None = None):
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError("--wandb was set but wandb is not installed. Please run: pip install wandb") from exc

    os.environ.setdefault("WANDB_MODE", "offline")

    name = args.wandb_run_name
    if name is None and default_name_prefix is not None:
        name = f"{default_name_prefix}_{datetime.now().strftime('%m%d_%H%M')}"

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=name,
        tags=args.wandb_tags,
        config=vars(args),
    )
    # Group panels into sections in this order: eval/, train/, time/
    wandb.define_metric("epoch")
    for key in ("eval/wer", "eval/cer", "train/loss", "train/lr",
                "time/data", "time/fwd", "time/bwd", "time/opt",
                "time/eval", "time/epoch"):
        wandb.define_metric(key, step_metric="epoch")
    return run


def wandb_log(run, metrics: dict):
    if run is not None:
        run.log(metrics)


def finish_wandb(run):
    if run is not None:
        run.finish()
