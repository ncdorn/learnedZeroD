"""Geometry + NN pipeline: chdir to tmp workspace, then run learned 0D steps."""

from __future__ import annotations

import importlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from learnedzerod.bundled_models import assert_junction_models_exist, vessel_model_dir_from_junction_dir
from learnedzerod.pipeline_nn import run_all_nn
from learnedzerod.presets import RUN_CONFIG
from learnedzerod.svzerod_runner import run_svzerod_forward
from learnedzerod._internal.zerod_calibration.bifurcation_splitting import (
    adjust_junction_boundaries_by_entrance_length_from_files,
    split_junctions_from_files,
)
from learnedzerod._internal.zerod_calibration.centerline_path_extraction import process_geometric_input
from learnedzerod._internal.zerod_calibration.file_io import get_paths
from learnedzerod._internal.zerod_calibration.geometric_params import extract_and_add_geometric_params
from learnedzerod._internal.zerod_calibration.run_config_canonical import canonical_run_config_for_data_paths


def _preload_util_for_dill_checkpoints() -> None:
    """Dill models saved from learn_lpns reference ``util.neural_network.nn_model``."""
    importlib.import_module("util.neural_network.nn_model")


_preload_util_for_dill_checkpoints()


def get_run_config_suffix_flags(
    normalize: bool,
    stenosis_off: bool,
    symmetric_loss: bool,
    clip_predictions: bool,
    penalty_off: bool,
) -> str:
    parts = []
    if normalize:
        parts.append("normalized")
    if stenosis_off:
        parts.append("stenosis_off")
    if symmetric_loss:
        parts.append("symmetric")
    if clip_predictions:
        parts.append("clip")
    if penalty_off:
        parts.append("penalty_off")
    return "_".join(parts) if parts else "base"


def build_args(
    set_name: str,
    geo_name: str,
    centerline_path: str,
    model_dir: str,
    trial_id: int = 0,
    verbose: bool = False,
) -> SimpleNamespace:
    """
    Match generate_zerod_inputs flag parity for ``stenosis_off_symmetric_gen_loss``.
    """
    run_config_suffix = RUN_CONFIG
    flag_suffix = get_run_config_suffix_flags(
        normalize=False,
        stenosis_off=True,
        symmetric_loss=True,
        clip_predictions=False,
        penalty_off=False,
    )
    canon = canonical_run_config_for_data_paths(run_config_suffix) or run_config_suffix
    if canon != flag_suffix:
        raise ValueError(
            f"Run config {run_config_suffix!r} flag mismatch (canonical {canon!r} vs {flag_suffix!r})"
        )

    return SimpleNamespace(
        set_name=set_name,
        geo_name=geo_name,
        junction_types=["BloodVesselJunction"],
        centerline_path=centerline_path,
        normalize=False,
        stenosis_off=True,
        symmetric_loss=True,
        clip_predictions=False,
        penalty_off=False,
        verbose=verbose,
        verbose_nn=verbose,
        NN_only=True,
        NN_vessel=True,
        model_dir=model_dir,
        trial_id=trial_id,
        no_redo=False,
    )


def run_learned_pipeline(
    *,
    zerod_json_path: str,
    centerline_vtp_path: str,
    output_dir: Path,
    set_name: str,
    svzerod_executable: str,
    geo_name: Optional[str] = None,
    keep_tmp: bool = True,
    output_filename: Optional[str] = None,
    verbose: bool = False,
) -> Path:
    """
    Run full pipeline with cwd = ``output_dir / tmp``.

    Returns path to the final JSON written under ``output_dir``.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = output_dir / "tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    geo_name = geo_name or Path(zerod_json_path).stem

    junction_dir = assert_junction_models_exist(set_name)
    vessel_dir = vessel_model_dir_from_junction_dir(junction_dir)
    for i in range(3):
        p = vessel_dir / f"rri_{set_name}_vessel_pred_{i}_model"
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing bundled vessel model: {p}. "
                "Copy bifurcations_EL_vessel_trial_0 dill files from learn_lpns results."
            )

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp)

        args = build_args(
            set_name=set_name,
            geo_name=geo_name,
            centerline_path=os.path.abspath(centerline_vtp_path),
            model_dir=str(junction_dir),
            verbose=verbose,
        )

        output_dir_rel = "data/zeroD"
        base_dir = os.path.join(output_dir_rel, set_name, RUN_CONFIG, geo_name)
        os.makedirs(base_dir, exist_ok=True)

        ml_inputs_base = os.path.join("data", "ml_inputs", set_name, RUN_CONFIG)

        geometry_variants, geometric_input_path, _, _, _, _, _, _ = get_paths(base_dir, args)

        with open(zerod_json_path, "r") as f:
            zerod_input = json.load(f)
        sim_params = zerod_input.setdefault("simulation_parameters", {})
        sim_params["output_all_cycles"] = True
        # Keep user-provided cycle count if present; only default when missing.
        sim_params.setdefault("number_of_cardiac_cycles", 1)
        os.makedirs(os.path.dirname(geometric_input_path), exist_ok=True)
        with open(geometric_input_path, "w") as f:
            json.dump(zerod_input, f, indent=4)

        geometric_centerline_input_path = geometric_input_path.replace(
            "geometric_input", "geometric_centerline_input"
        )
        process_geometric_input(
            args.centerline_path,
            geometric_input_path,
            geometric_centerline_input_path,
        )

        bifurcations_geometric_input_path = geometry_variants["bifurcations"]["geometric_input"]
        split_junctions_from_files(
            geometric_centerline_input_path,
            args.centerline_path,
            bifurcations_geometric_input_path,
        )

        bifurcations_EL_geometric_input_path = geometry_variants["bifurcations_EL"]["geometric_input"]
        adjust_junction_boundaries_by_entrance_length_from_files(
            bifurcations_geometric_input_path,
            args.centerline_path,
            bifurcations_EL_geometric_input_path,
            verbose=args.verbose,
        )

        for geo_variant_name, geo_variant_paths in geometry_variants.items():
            if geo_variant_name == "bifurcations_EL":
                bif_in = geometry_variants["bifurcations"]["geometric_input"]
                if not os.path.exists(bif_in):
                    continue
            if geo_variant_name == "original":
                continue
            variant_geometric_input = geo_variant_paths["geometric_input"]
            if not os.path.exists(variant_geometric_input):
                continue
            if geo_variant_name == "bifurcations_EL":
                extract_and_add_geometric_params(
                    args.centerline_path,
                    geometry_variants["bifurcations"]["geometric_input"],
                    variant_geometric_input,
                    el_adjusted_geometric_input_path=variant_geometric_input,
                )
            else:
                extract_and_add_geometric_params(
                    args.centerline_path,
                    variant_geometric_input,
                    variant_geometric_input,
                )

        for _name, _paths in (
            ("bifurcations", geometry_variants["bifurcations"]),
            ("bifurcations_EL", geometry_variants["bifurcations_EL"]),
        ):
            gin = _paths["geometric_input"]
            if not os.path.isfile(gin):
                continue
            gcsv = gin.replace("_geometric_input.json", "_geometric_results.csv")
            run_svzerod_forward(svzerod_executable, gin, gcsv, verbose=verbose)

        run_all_nn(
            args,
            geometry_variants,
            base_dir,
            ml_inputs_base,
            RUN_CONFIG,
            str(vessel_dir),
        )

        final_src = os.path.join(base_dir, "bifurcations_EL_NN_JunctionAndVessel.json")
        if not os.path.isfile(final_src):
            raise FileNotFoundError(
                f"Expected output not found: {final_src}. "
                "Check NN logs; junction/vessel models must exist."
            )
        final_name = output_filename or f"{geo_name}_learned.json"
        final_dst = output_dir / final_name
        shutil.copy2(final_src, final_dst)
        return final_dst
    finally:
        os.chdir(old_cwd)
        if not keep_tmp and tmp.exists():
            shutil.rmtree(tmp)
