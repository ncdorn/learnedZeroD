"""CLI entry: ``learned-zerod``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from learnedzerod.pipeline import run_learned_pipeline
from learnedzerod.presets import PRESET_TO_SET_NAME


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Learn junction and vessel R/L/S from Richter 0D JSON + centerline VTP."
    )
    parser.add_argument(
        "--anatomy",
        choices=sorted(PRESET_TO_SET_NAME.keys()),
        required=True,
        help="Anatomy preset (selects VMR set_name and bundled model folder).",
    )
    parser.add_argument(
        "--zerod-json",
        type=Path,
        required=True,
        help="Path to Richter-style 0D input JSON.",
    )
    parser.add_argument(
        "--centerline-vtp",
        type=Path,
        required=True,
        help="Path to centerline VTP (e.g. unsteady_soln.vtp).",
    )
    parser.add_argument(
        "--svzerod",
        type=Path,
        required=True,
        help="Path to svZeroDSolver executable (runs geometric 0D for flow-split CSVs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for tmp/ intermediates and final JSON.",
    )
    parser.add_argument(
        "--geo-name",
        default=None,
        help="Case folder name (default: stem of --zerod-json).",
    )
    parser.add_argument(
        "--output-filename",
        default=None,
        help="Final JSON filename inside output-dir (default: [geo_name]_learned.json).",
    )
    parser.add_argument(
        "--no-keep-tmp",
        action="store_true",
        help="Remove output-dir/tmp after a successful run.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args(argv)

    zj = args.zerod_json.resolve()
    cl = args.centerline_vtp.resolve()
    od = args.output_dir.resolve()
    sz = args.svzerod.resolve()
    if not zj.is_file():
        print(f"error: --zerod-json not found: {zj}", file=sys.stderr)
        return 2
    if not cl.is_file():
        print(f"error: --centerline-vtp not found: {cl}", file=sys.stderr)
        return 2
    if not sz.is_file():
        print(f"error: --svzerod not found: {sz}", file=sys.stderr)
        return 2
    if not os.access(sz, os.X_OK):
        print(f"error: --svzerod is not executable: {sz}", file=sys.stderr)
        return 2

    set_name = PRESET_TO_SET_NAME[args.anatomy]
    try:
        out = run_learned_pipeline(
            zerod_json_path=str(zj),
            centerline_vtp_path=str(cl),
            output_dir=od,
            set_name=set_name,
            svzerod_executable=str(sz),
            geo_name=args.geo_name,
            keep_tmp=not args.no_keep_tmp,
            output_filename=args.output_filename,
            verbose=args.verbose,
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        if args.verbose:
            raise
        return 1

    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
