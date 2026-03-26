"""Smoke tests: imports and presets (no NN weights required)."""

from pathlib import Path

import pytest


def test_presets():
    from learnedzerod.presets import PRESET_TO_SET_NAME, RUN_CONFIG

    assert PRESET_TO_SET_NAME["aortic"] == "VMR_rigid_aorta_adults_all"
    assert RUN_CONFIG == "stenosis_off_symmetric_gen_loss"


def test_bundled_paths_resolve():
    from learnedzerod import bundled_models

    d = bundled_models.junction_model_dir("VMR_rigid_aorta_adults_all")
    assert d.name == "bifurcations_EL_trial_0"
    assert "models" in str(d)


def test_package_version():
    import learnedzerod

    assert hasattr(learnedzerod, "__version__")


def test_cli_module_importable():
    import learnedzerod.cli

    assert hasattr(learnedzerod.cli, "main")


def test_sanitize_json_no_nan_token():
    import json
    import math

    from learnedzerod._internal.json_sanitize import sanitize_for_svzerod_json

    raw = {"L": [float("nan"), 1.0], "x": None, "y": float("inf")}
    clean = sanitize_for_svzerod_json(raw)
    s = json.dumps(clean)
    assert "NaN" not in s
    assert "Infinity" not in s
    assert clean["L"][0] == 0.0
    assert clean["x"] == 0.0
    assert clean["y"] == 0.0


def test_util_shim_for_dill():
    """learn_lpns checkpoints reference util.neural_network.nn_model.NeuralNet."""
    import util.neural_network.nn_model as nm

    assert hasattr(nm, "NeuralNet")
    assert hasattr(nm, "predict")


def test_models_readme_exists():
    import learnedzerod

    root = Path(learnedzerod.__file__).resolve().parent / "models" / "README.md"
    assert root.is_file()


def test_svzerod_runner_rejects_missing_exe():
    from learnedzerod.svzerod_runner import run_svzerod_forward

    with pytest.raises(FileNotFoundError, match="svzerod executable not found"):
        run_svzerod_forward(
            "/nonexistent/svZeroDSolver",
            "/nonexistent/in.json",
            "/nonexistent/out.csv",
        )
