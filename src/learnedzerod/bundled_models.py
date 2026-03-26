"""Absolute paths to bundled junction (and derived vessel) NN checkpoints."""

from __future__ import annotations

from pathlib import Path

from learnedzerod.presets import RUN_CONFIG


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def junction_model_dir(set_name: str) -> Path:
    """Directory containing ``rri_{set_name}_pred_{0,1,2}_model`` (trial 0, bifurcations_EL)."""
    return _package_dir() / "models" / set_name / RUN_CONFIG / "bifurcations_EL_trial_0"


def vessel_model_dir_from_junction_dir(junction_dir: Path) -> Path:
    """Sibling ``bifurcations_EL_vessel_trial_0`` matching CV layout."""
    base = junction_dir.parent
    name = junction_dir.name
    if "_trial_" in name:
        vessel_name = name.replace("_trial_", "_vessel_trial_", 1)
    else:
        vessel_name = name.replace("bifurcations_EL", "bifurcations_EL_vessel", 1)
    return base / vessel_name


def assert_junction_models_exist(set_name: str) -> Path:
    d = junction_model_dir(set_name)
    for i in range(3):
        p = d / f"rri_{set_name}_pred_{i}_model"
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing bundled junction model: {p}. "
                "Copy trial-0 dill checkpoints from learn_lpns results into this path."
            )
    return d
