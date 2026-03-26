"""Junction + vessel NN inference (vendored logic from generate_zerod_inputs Step 3.6–3.7)."""

from __future__ import annotations

import importlib
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd

importlib.import_module("util.neural_network.nn_model")

from learnedzerod._internal.data_processing.data_dict_from_csvs import (
    filter_features_from_array,
    get_default_include_features_vessel,
)
from learnedzerod._internal.data_processing.inputs_from_0d_config import (
    compute_junction_flow_splits,
    load_junction_geometric_features,
    load_vessel_geometric_features,
)
from learnedzerod._internal.neural_network.nn_predict import predict
from learnedzerod._internal.neural_network.nn_util import dill_load
from learnedzerod._internal.json_sanitize import sanitize_for_svzerod_json
from learnedzerod._internal.zerod_calibration.bifurcation_splitting import (
    convert_el_normal_junctions_to_blood_vessel_junction,
)


def _raise_if_predictions_not_finite(name: str, arr: np.ndarray) -> None:
    a = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(a)):
        bad = np.flatnonzero(~np.isfinite(a))
        preview = bad[:16].tolist()
        raise ValueError(
            f"Non-finite NN predictions in {name}: {len(bad)} bad value(s), "
            f"first indices {preview}"
        )


def run_junction_nn(
    args: SimpleNamespace,
    geo_variant_name: str,
    geometry_variants: Dict[str, Any],
    base_dir: str,
    ml_inputs_base: str,
    run_config_suffix: Optional[str],
) -> None:
    variant_geometric_input = geometry_variants[geo_variant_name]["geometric_input"]
    nn_output_path = os.path.join(base_dir, f"{geo_variant_name}_NN_BloodVesselJunction.json")

    if not os.path.exists(variant_geometric_input):
        raise FileNotFoundError(f"Geometric input not found: {variant_geometric_input}")

    with open(variant_geometric_input, "r") as f:
        nn_config = json.load(f)

    if geo_variant_name == "bifurcations":
        convert_el_normal_junctions_to_blood_vessel_junction(nn_config)

    csv_path = os.path.join(ml_inputs_base, geo_variant_name, args.geo_name, "geometric_features.csv")
    print("  Extracting features directly from geometric input")
    X_full, feature_names_full, junction_names, outlet_primary_names = load_junction_geometric_features(
        variant_geometric_input,
        require_two_outlets=True,
        verbose=True,
    )
    geometric_results_path = variant_geometric_input.replace(
        "_geometric_input.json", "_geometric_results.csv"
    )
    if not os.path.isfile(geometric_results_path):
        raise FileNotFoundError(
            f"Geometric 0D results CSV not found: {geometric_results_path}. "
            "The pipeline must run svzerod on the geometric JSON first (--svzerod)."
        )
    flow_splits = compute_junction_flow_splits(variant_geometric_input, geometric_results_path)
    flow_split_col = []
    for i, jname in enumerate(junction_names):
        if jname not in flow_splits:
            raise ValueError(
                f"No flow-split row for junction {jname!r} in {geometric_results_path}"
            )
        (out0_name, out1_name), (fs0, fs1) = flow_splits[jname]
        primary = outlet_primary_names[i]
        val = (
            fs0
            if primary == out0_name
            else (fs1 if primary == out1_name else float("nan"))
        )
        if not np.isfinite(val):
            raise ValueError(
                f"Non-finite flow_split for junction {jname!r} (primary outlet {primary!r}) "
                f"from {geometric_results_path}"
            )
        flow_split_col.append(val)
    X_full = np.column_stack([X_full, flow_split_col])
    feature_names_full = feature_names_full + ["flow_split"]
    flow_split_arr = np.asarray(flow_split_col, dtype=float)
    if not np.all(flow_split_arr > 0):
        bad = np.flatnonzero(~(flow_split_arr > 0))
        raise ValueError(
            f"flow_split must be positive for NN features; bad row indices {bad[:16].tolist()}"
        )
    flow_split_inv = 100.0 / flow_split_arr
    X_full = np.column_stack([X_full, flow_split_inv])
    feature_names_full = feature_names_full + ["flow_split_inv"]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame(X_full, columns=feature_names_full).to_csv(csv_path, index=False)

    X, feature_names = filter_features_from_array(X_full, feature_names_full, remap_tortuosity=True)
    if len(X) == 0:
        raise ValueError("No junctions found in geometric features")

    norm_suffix = "_normalized" if args.normalize else ""
    if args.normalize:
        raise NotImplementedError("Normalized NN inference requires jax norm pkls; use raw models.")

    X_for_nn = X
    X_jax = jnp.array(X_for_nn, dtype=jnp.float32)

    model_dir = args.model_dir
    model_base_name = f"rri_{args.set_name}_pred"
    model_paths = [
        os.path.join(model_dir, f"{model_base_name}_0_model"),
        os.path.join(model_dir, f"{model_base_name}_1_model"),
        os.path.join(model_dir, f"{model_base_name}_2_model"),
    ]
    for mp in model_paths:
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Model not found: {mp}")

    raw_predictions = []
    for i, model_path in enumerate(model_paths):
        print(f"      Loading model {i + 1}/3: {model_path}")
        model = dill_load(model_path)
        use_leaky = getattr(model, "use_leaky_relu", False)
        pred = predict(X_jax, model.weights, use_leaky)
        raw_predictions.append(np.array(pred).flatten())

    predictions = raw_predictions
    pred_R = np.array(predictions[0])
    pred_S = np.zeros_like(pred_R) if getattr(args, "stenosis_off", False) else np.array(predictions[1])
    pred_L = np.array(predictions[2])
    _raise_if_predictions_not_finite("junction R_poiseuille", pred_R)
    _raise_if_predictions_not_finite("junction stenosis_coefficient", pred_S)
    _raise_if_predictions_not_finite("junction L", pred_L)

    if len(pred_R) != len(X):
        raise ValueError("Prediction array size mismatch vs feature rows")

    primary_outlet_to_row = {}
    junction_name_to_row_indices = {}
    for row_idx, (junc_name, pout_name) in enumerate(zip(junction_names, outlet_primary_names)):
        primary_outlet_to_row[(junc_name, pout_name)] = row_idx
        if junc_name not in junction_name_to_row_indices:
            junction_name_to_row_indices[junc_name] = []
        junction_name_to_row_indices[junc_name].append(row_idx)

    vessels = nn_config.get("vessels", [])
    vessel_id_to_name = {v.get("vessel_id"): v.get("vessel_name", "") for v in vessels}

    for junc in nn_config.get("junctions", []):
        junc_name = junc.get("junction_name", "")
        if junc_name not in junction_name_to_row_indices:
            if junc.get("junction_type") == "BloodVesselJunction":
                if "junction_values" not in junc:
                    outlet_vessel_ids = junc.get("outlet_vessels", [])
                    num_outlets = len(outlet_vessel_ids)
                    junc["junction_values"] = {
                        "R_poiseuille": [0.0] * num_outlets,
                        "stenosis_coefficient": [0.0] * num_outlets,
                        "L": [0.0] * num_outlets,
                    }
            continue

        outlet_vessel_ids = junc.get("outlet_vessels", [])
        if len(outlet_vessel_ids) != 2:
            continue

        outlet_vessel_names = [vessel_id_to_name.get(vid, "") for vid in outlet_vessel_ids]
        if "junction_values" not in junc:
            junc["junction_values"] = {}

        R_values = [0.0] * len(outlet_vessel_ids)
        S_values = [0.0] * len(outlet_vessel_ids)
        L_values = [0.0] * len(outlet_vessel_ids)

        for file_idx, (vid, vname) in enumerate(zip(outlet_vessel_ids, outlet_vessel_names)):
            if "connector" in vname and "connectorEL" not in vname:
                continue
            row_idx = primary_outlet_to_row.get((junc_name, vname))
            if row_idx is not None:
                expected_vid = int(X_full[row_idx, 0])
                if expected_vid != vid:
                    raise ValueError(f"Vessel ID mismatch {junc_name}/{vname}")
                R_values[file_idx] = float(pred_R[row_idx])
                S_values[file_idx] = float(pred_S[row_idx])
                L_values[file_idx] = float(pred_L[row_idx])
            else:
                other_outlet = [on for on in outlet_vessel_names if on != vname]
                fallback_row = None
                if other_outlet:
                    fallback_row = primary_outlet_to_row.get((junc_name, other_outlet[0]))
                if fallback_row is not None:
                    R_values[file_idx] = float(pred_R[fallback_row])
                    S_values[file_idx] = float(pred_S[fallback_row])
                    L_values[file_idx] = float(pred_L[fallback_row])

        junc["junction_values"]["R_poiseuille"] = R_values
        junc["junction_values"]["stenosis_coefficient"] = S_values
        junc["junction_values"]["L"] = L_values

    with open(nn_output_path, "w") as f:
        json.dump(sanitize_for_svzerod_json(nn_config), f, indent=4)
    print(f"      ✓ Junction NN saved to {nn_output_path}")


def run_vessel_nn(
    args: SimpleNamespace,
    geo_variant_name: str,
    geometry_variants: Dict[str, Any],
    base_dir: str,
    run_config_suffix: Optional[str],
    vessel_model_dir: str,
) -> None:
    nn_output_path = os.path.join(base_dir, f"{geo_variant_name}_NN_BloodVesselJunction.json")
    if not os.path.exists(nn_output_path):
        print(f"  Skipping vessel NN: missing {nn_output_path}")
        return

    nn_jv_path = os.path.join(base_dir, f"{geo_variant_name}_NN_JunctionAndVessel.json")
    nn_vessel_only_path = os.path.join(base_dir, f"{geo_variant_name}_NN_VesselOnly.json")

    model_dir_basename = os.path.basename(getattr(args, "model_dir", "") or "")
    if "_trial_" in model_dir_basename:
        _first = model_dir_basename.split("_trial_")[0]
        _model_variant = _first.replace("_normalized", "")
    else:
        _model_variant = None
    if _model_variant is not None and geo_variant_name != _model_variant:
        return

    with open(nn_output_path, "r") as f:
        jv_config = json.load(f)

    variant_geometric_input = geometry_variants[geo_variant_name]["geometric_input"]
    X_v, feat_names_v, vessel_ids, vessel_names = load_vessel_geometric_features(
        variant_geometric_input, verbose=getattr(args, "verbose", False)
    )
    if len(X_v) == 0:
        print(f"      No non-connector vessels for {geo_variant_name}")
        return

    X_v, feat_names_v = filter_features_from_array(
        X_v,
        feat_names_v,
        include_features=get_default_include_features_vessel(),
        remap_tortuosity=False,
    )

    if args.normalize:
        raise NotImplementedError("Normalized vessel NN not implemented in learnedZeroD yet.")

    X_v_nn = np.array(X_v, dtype=np.float64)
    X_v_jax = jnp.array(X_v_nn, dtype=jnp.float32)

    model_paths = [
        os.path.join(vessel_model_dir, f"rri_{args.set_name}_vessel_pred_{i}_model") for i in range(3)
    ]
    for mp in model_paths:
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Vessel model not found: {mp}")

    raw_predictions_v = []
    for mp in model_paths:
        model = dill_load(mp)
        use_leaky_v = getattr(model, "use_leaky_relu", False)
        pred = predict(X_v_jax, model.weights, use_leaky_v)
        raw_predictions_v.append(np.array(pred).flatten())

    pred_R_v = np.array(raw_predictions_v[0])
    pred_S_v = np.array(raw_predictions_v[1])
    pred_L_v = np.array(raw_predictions_v[2])
    if getattr(args, "stenosis_off", False):
        pred_S_v = np.zeros_like(pred_R_v)
    _raise_if_predictions_not_finite("vessel R_poiseuille", pred_R_v)
    _raise_if_predictions_not_finite("vessel stenosis_coefficient", pred_S_v)
    _raise_if_predictions_not_finite("vessel L", pred_L_v)

    vessel_id_to_row = {vid: i for i, vid in enumerate(vessel_ids)}
    for v in jv_config.get("vessels", []):
        vname = (v.get("vessel_name") or "").lower()
        if "connector" in vname:
            continue
        vid = v.get("vessel_id")
        row = vessel_id_to_row.get(vid)
        if row is None:
            continue
        z = dict(v.get("zero_d_element_values") or {})
        z["R_poiseuille"] = float(pred_R_v[row])
        z["stenosis_coefficient"] = float(pred_S_v[row])
        z["L"] = float(pred_L_v[row])
        v["zero_d_element_values"] = z

    with open(nn_jv_path, "w") as f:
        json.dump(sanitize_for_svzerod_json(jv_config), f, indent=4)
    print(f"      ✓ Junction+Vessel NN saved to {nn_jv_path}")

    with open(variant_geometric_input, "r") as f:
        vessel_only_config = json.load(f)
    for v in vessel_only_config.get("vessels", []):
        vname = (v.get("vessel_name") or "").lower()
        if "connector" in vname:
            continue
        vid = v.get("vessel_id")
        row = vessel_id_to_row.get(vid)
        if row is None:
            continue
        if "zero_d_element_values" not in v:
            v["zero_d_element_values"] = {}
        v["zero_d_element_values"]["R_poiseuille"] = float(pred_R_v[row])
        v["zero_d_element_values"]["stenosis_coefficient"] = float(pred_S_v[row])
        v["zero_d_element_values"]["L"] = float(pred_L_v[row])
    with open(nn_vessel_only_path, "w") as f:
        json.dump(sanitize_for_svzerod_json(vessel_only_config), f, indent=4)


def run_all_nn(
    args: SimpleNamespace,
    geometry_variants: Dict[str, Any],
    base_dir: str,
    ml_inputs_base: str,
    run_config_suffix: Optional[str],
    vessel_model_dir: str,
) -> None:
    for geo_variant_name in ("bifurcations", "bifurcations_EL"):
        if geo_variant_name not in geometry_variants:
            continue
        bif_in = geometry_variants["bifurcations"]["geometric_input"]
        if geo_variant_name == "bifurcations_EL" and not os.path.exists(bif_in):
            continue
        run_junction_nn(
            args, geo_variant_name, geometry_variants, base_dir, ml_inputs_base, run_config_suffix
        )

    if getattr(args, "NN_vessel", False):
        for geo_variant_name in ("bifurcations", "bifurcations_EL"):
            if geo_variant_name not in geometry_variants:
                continue
            run_vessel_nn(
                args,
                geo_variant_name,
                geometry_variants,
                base_dir,
                run_config_suffix,
                vessel_model_dir,
            )
