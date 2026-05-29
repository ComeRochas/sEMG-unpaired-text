from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml


def apply_yaml_defaults(parser: argparse.ArgumentParser, config_path: str | None) -> None:
    """Merge YAML values into the parser as defaults.

    Precedence: CLI > YAML > script defaults.
    Mutates `parser` in place via `set_defaults`. No-op if `config_path` is None.
    """
    if not config_path:
        return
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with path.open("r") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    parser.set_defaults(**cfg)


def parse_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Two-pass parse so `--config foo.yaml` overrides script defaults but not CLI args."""
    pre, _ = parser.parse_known_args()
    apply_yaml_defaults(parser, getattr(pre, "config", None))
    return parser.parse_args()


def setup_stdout_logging(level: int = logging.INFO) -> None:
    """Send INFO-level logs to stdout (default logging goes to stderr)."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(level)
