"""
Recursively clean objects for ``json.dump`` + strict C++ JSON parsers (svZeroDSolver).

Python's default JSON encoder writes ``NaN`` / ``Infinity`` as invalid JSON tokens.
``None`` is encoded as ``null``; this codebase historically maps nulls to ``0.0`` for nlohmann.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def sanitize_for_svzerod_json(obj: Any) -> Any:
    if obj is None:
        return 0.0
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_svzerod_json(obj.tolist())
    if isinstance(obj, dict):
        return {k: sanitize_for_svzerod_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_svzerod_json(v) for v in obj]
    return obj
