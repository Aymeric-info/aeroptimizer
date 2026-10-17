#!/usr/bin/env python
"""Entry point: python scripts/run.py configs/mbb_2d.yaml"""

from __future__ import annotations

import argparse

from aeroptimizer.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_config(args.config)  # strict validation (C-CFG-01)
    raise NotImplementedError(f"phase 2: Problem.from_config(cfg) + optim.loop.run — case={cfg.case!r}")


if __name__ == "__main__":
    main()
