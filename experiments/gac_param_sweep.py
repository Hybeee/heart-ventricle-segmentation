import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndimage
import re
import json

import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

BASELINE = {
    "curvature_scaling": 4.0,
    "advection_scaling": 1.0,
    "propagation_scaling": -0.3,
    "sigma": 2.0,
    "num_iterations": 600,
    "max_rms_error": 0.0001,
}
PARAM_COLORS = {
    "baseline": "0.6 0.6 0.6",
    "curvature_scaling": "0.85 0.1 0.1",
    "advection_scaling": "0.1 0.45 0.85",
    "propagation_scaling": "0.15 0.65 0.25",
    "sigma": "0.9 0.6 0.0",
}

def _get_bbox_with_padding(mask_np, padding_voxels):
    coords = np.argwhere(mask_np)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1

    pad = np.asarray(padding_voxels)
    mins = np.maximum(mins - pad, 0)
    maxs = np.minimum(maxs + pad, np.array(mask_np.shape))

    return mins, maxs

def _run_gac(signed_distance, initial_level_set, params):
    signed_distance_np = sitk.GetArrayFromImage(signed_distance)
    edge_potential_np = 1.0 - np.exp(-np.abs(signed_distance_np) / params["sigma"])
    edge_potential = sitk.GetImageFromArray(edge_potential_np.astype(np.float32))
    edge_potential.CopyInformation(initial_level_set)

    active_contour = sitk.GeodesicActiveContourLevelSetImageFilter()
    active_contour.SetCurvatureScaling(params["curvature_scaling"])
    active_contour.SetAdvectionScaling(params["advection_scaling"])
    active_contour.SetPropagationScaling(params["propagation_scaling"])
    active_contour.SetMaximumRMSError(params["max_rms_error"])
    active_contour.SetNumberOfIterations(params["num_iterations"])

    final_level_set = active_contour.Execute(initial_level_set, edge_potential)
    final_level_set_np = sitk.GetArrayFromImage(final_level_set)
    roi_binary_mask = (final_level_set_np < 0).astype(np.uint8)

    return roi_binary_mask

def _process_patient(patient_dir, output_dir, param_sets: dict):
    orig_mask_sitk, mask = utils.scan_to_np_array(scan_path=os.path.join(patient_dir, "final_mask_nip.seg.nrrd"), return_sitk=True)
    nnunet_sitk, nnunet_mask = utils.scan_to_np_array(scan_path=os.path.join(patient_dir, "nnunet_mask.seg.nrrd"), return_sitk=True)

    padding_mm = 5.0
    spacing_xyz = orig_mask_sitk.GetSpacing()
    padding_voxels = [
        int(np.ceil(padding_mm / spacing_xyz[2])),
        int(np.ceil(padding_mm / spacing_xyz[1])),
        int(np.ceil(padding_mm / spacing_xyz[0]))
    ]
    mins, maxs = _get_bbox_with_padding(
        mask_np=nnunet_mask, padding_voxels=padding_voxels
    )
    z_min, y_min, x_min = mins
    z_max, y_max, x_max = maxs

    roi_index = [int(x_min), int(y_min), int(z_min)]
    roi_size = [int(x_max - x_min), int(y_max - y_min), int(z_max - z_min)]

    orig_mask_sitk_roi = sitk.RegionOfInterest(orig_mask_sitk, roi_size, roi_index)
    nnunet_sitk_roi = sitk.RegionOfInterest(nnunet_sitk, roi_size, roi_index)

    signed_distance = sitk.SignedMaurerDistanceMap(
        orig_mask_sitk_roi,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True
    )
    nnunet_signed_distance = sitk.SignedMaurerDistanceMap(
        nnunet_sitk_roi,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True,
    )
    initial_level_set = sitk.Cast(nnunet_signed_distance, sitk.sitkFloat32)

    roi_binary_mask = _run_gac(
        signed_distance=signed_distance,
        initial_level_set=initial_level_set,
        params=BASELINE
    )
    final_binary_mask = np.zeros_like(mask, dtype=np.uint8)
    final_binary_mask[z_min:z_max, y_min:y_max, x_min:x_max] = roi_binary_mask
    name = "baseline"
    utils.save_data(
        data=final_binary_mask,
        ref_sitk=orig_mask_sitk,
        output_dir=output_dir,
        name=name,
        is_mask=True,
        color=PARAM_COLORS['baseline'],
        segment_name=name
    )

    for param_name in param_sets.keys():
        curr_output_dir = os.path.join(output_dir, param_name)
        os.makedirs(curr_output_dir, exist_ok=True)

        params = BASELINE.copy()

        param_values = param_sets[param_name]
        start = time.time()
        for param_value in param_values:
            params[param_name] = param_value

            roi_binary_mask = _run_gac(
                signed_distance=signed_distance,
                initial_level_set=initial_level_set,
                params=params
            )

            final_binary_mask = np.zeros_like(mask, dtype=np.uint8)
            final_binary_mask[z_min:z_max, y_min:y_max, x_min:x_max] = roi_binary_mask
            name = f"{param_name}_{str(param_value)}"
            utils.save_data(
                data=final_binary_mask,
                ref_sitk=orig_mask_sitk,
                output_dir=curr_output_dir,
                name=name,
                is_mask=True,
                color=PARAM_COLORS[param_name],
                segment_name=name
            )
        end = time.time()
        print(f"\tSwept {param_name} in {(end-start):.4f}s")

def _build_param_sets():
    param_sets = {
        "curvature_scaling": np.geomspace(0.5, 80, num=9),
        "advection_scaling": np.geomspace(0.5, 80, num=9),
        "propagation_scaling": np.append(-np.geomspace(0.5, 60, num=8), [0.0, 0.3]),
        "sigma": np.geomspace(0.25, 500, num=7)
    }

    return {k: sorted(v.tolist()) for k, v in param_sets.items()}

def _sweep_params(data_dir, output_dir):

    param_sets = _build_param_sets()

    print("==============")
    print("PARAM SET")
    for k, v in param_sets.items():
        print(f"\t{k}: {v}")
    print("==============")

    for patient_id in sorted(os.listdir(data_dir)):
        print(f"Processing {patient_id}...")
        patient_dir = os.path.join(data_dir, patient_id)
        curr_output_dir = os.path.join(output_dir, patient_id)
        os.makedirs(curr_output_dir, exist_ok=True)

        _process_patient(patient_dir, curr_output_dir, param_sets)

def _get_dice_score(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()

    return (2.0 * intersection) / total if total > 0 else 1.0

def _flag_low_dice(output_dir, dice_threshold):
    baseline_mask = utils.scan_to_np_array(os.path.join(output_dir, "baseline.seg.nrrd"))

    param_names = ["curvature_scaling", "advection_scaling", "propagation_scaling", "sigma"]

    records = []

    for param_name in param_names:
        param_dir = os.path.join(output_dir, param_name)
        for mask_res_name in sorted(os.listdir(param_dir)):
            mask_res = utils.scan_to_np_array(os.path.join(param_dir, mask_res_name))
            dice = _get_dice_score(baseline_mask, mask_res)

            value = float(mask_res_name[len(param_name) + 1:-len(".seg.nrrd")])
            
            records.append({
                "param_name": param_name,
                "value": value,
                "dice": float(dice),
                "passed": bool(dice >= dice_threshold)
            })

    out_path = os.path.join(output_dir, "dice_results.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

def _get_param_interval(records, param_name, baseline_value, dice_threshold):
    sub = sorted(
        (r["value"], r["dice"]) for r in records if r["param_name"] == param_name
    )
    values = [v for v, _ in sub]
    dices = [d for _, d in sub]

    if not values:
        return None, None
    
    baseline_idx = np.searchsorted(values, baseline_value)

    lower = values[0]
    for i in range(baseline_idx - 1, -1, -1):
        if dices[i] < dice_threshold:
            lower = values[i + 1] if i + 1 <= baseline_idx - 1 else baseline_value
            break
    
    upper = values[-1]
    for i in range(baseline_idx, len(values)):
        if dices[i] < dice_threshold:
            upper = values[i - 1] if i - 1 >= baseline_idx else baseline_value
            break
    
    return lower, upper

def _load_dice_records(patient_output_dir):
    path = os.path.join(patient_output_dir, "dice_results.json")
    with open(path) as f:
        return json.load(f)

def _get_all_param_intervals(patient_output_dir, param_names, baseline=BASELINE, dice_threshold=0.95):
    records = _load_dice_records(patient_output_dir)

    intervals = {}
    for param_name in param_names:
        lower, upper = _get_param_interval(records, param_name, baseline[param_name], dice_threshold)
        if lower is None or upper is None:
            print(f"WARNING: {patient_output_dir} has no interval data for {param_name}")
            continue
        intervals[param_name] = [lower, upper]
    
    return intervals

def main():
    data_dir = os.path.join(ROOT_DIR, "pipeline_output")
    output_dir = os.path.join(ROOT_DIR, "gac_param_sweep_output")

    # _sweep_params(data_dir, output_dir)

    skip_patient_ids = [
        "patient_0002",
        "patient_0006",
        "patient_0008",
        "patient_0009",
        "patient_0010",
        "patient_0012",
        "patient_0013",
        "patient_0017", # NAGYON ERZEKENY A PROPAGATION SCALINGRE ES EGYEBKENT A TOBBIRE IS
        "patient_0019",
        "patient_0020",
        "patient_0021",
        "patient_0022"
    ]

    # for patient_id in sorted(os.listdir(output_dir)):
    #     print(patient_id)
    #     if patient_id in skip_patient_ids:
    #         continue

    #     _flag_low_dice(
    #         output_dir=os.path.join(output_dir, patient_id),
    #         dice_threshold=0.95
    #     )

    #     if patient_id == "patient_0018":
    #         break

    param_names = ["curvature_scaling", "advection_scaling", "propagation_scaling", "sigma"]
    all_intervals = {p: [] for p in param_names}
    for patient_id in sorted(os.listdir(output_dir)):
        if patient_id in skip_patient_ids:
            continue
        
        patient_dir = os.path.join(output_dir, patient_id)
        intervals = _get_all_param_intervals(patient_dir, param_names)
        for param_name, (lower, upper) in intervals.items():
            all_intervals[param_name].append((lower, upper))
    
    for param_name, bounds in all_intervals.items():
        arr = np.array(bounds)
        global_lower = arr[:, 0].max()
        global_upper = arr[:, 1].min()
        print(f"{param_name.upper()}: {global_lower} - {global_upper}")

if __name__ == "__main__":
    main()