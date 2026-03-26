"""Run svZeroDSolver on geometric JSON to produce results CSV for flow-split features."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run_svzerod_forward(
    svzerod_executable: str | os.PathLike[str],
    input_json: str | os.PathLike[str],
    output_csv: str | os.PathLike[str],
    *,
    verbose: bool = False,
) -> None:
    """
    Execute ``svzerod input.json output.csv`` (same convention as svZeroDPlus CLI).

    Raises:
        FileNotFoundError: executable or input JSON missing.
        RuntimeError: non-zero exit or CSV not created.
    """
    exe = Path(svzerod_executable).resolve()
    inp = Path(input_json).resolve()
    out = Path(output_csv).resolve()

    if not exe.is_file():
        raise FileNotFoundError(f"svzerod executable not found: {exe}")
    if not os.access(exe, os.X_OK):
        raise FileNotFoundError(f"svzerod is not executable: {exe}")
    if not inp.is_file():
        raise FileNotFoundError(f"Geometric input JSON not found: {inp}")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    cmd = [str(exe), str(inp), str(out)]
    if verbose:
        print(f"  Running: {' '.join(cmd)}")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"svzerod failed with exit code {proc.returncode}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )
    if not out.is_file():
        raise RuntimeError(f"svzerod did not write output CSV: {out}")
