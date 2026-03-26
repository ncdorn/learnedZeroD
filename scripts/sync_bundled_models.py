#!/usr/bin/env python3
"""
Copy trial-0 junction and vessel NN checkpoints from a learn_lpns results tree into
``src/learnedzerod/models/`` for packaging.

Example::

    python scripts/sync_bundled_models.py --learn-lpns-root /path/to/learn_lpns
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Repo root: scripts/ -> learnedZeroD/
_ROOT = Path(__file__).resolve().parent.parent
_DST_MODELS = _ROOT / "src" / "learnedzerod" / "models"

# Same mapping as learnedzerod.presets (avoid import before install)
_SET_NAMES = (
    "VMR_rigid_aorta_adults_all",
    "VMR_abdo",
    "VMR_pulmo_healthy",
    "VMR_all",
)
_RUN_CONFIG = "stenosis_off_symmetric_gen_loss"
_SUBDIRS = ("bifurcations_EL_trial_0", "bifurcations_EL_vessel_trial_0")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--learn-lpns-root",
        type=Path,
        required=True,
        help="Root of the learn_lpns repository (contains results/models/).",
    )
    args = p.parse_args()
    src_base = (args.learn_lpns_root / "results" / "models").resolve()
    if not src_base.is_dir():
        print(f"error: not a directory: {src_base}", file=sys.stderr)
        return 2

    copied = 0
    for set_name in _SET_NAMES:
        for sub in _SUBDIRS:
            src = src_base / set_name / _RUN_CONFIG / sub
            if not src.is_dir():
                print(f"  skip (missing): {src}")
                continue
            dst = _DST_MODELS / set_name / _RUN_CONFIG / sub
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  copied -> {dst}")
            copied += 1

    if copied == 0:
        print(
            "warning: no model directories found under results/models/.../stenosis_off_symmetric_gen_loss/",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
