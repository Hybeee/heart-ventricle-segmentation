import SimpleITK as sitk
import numpy as np

import os
import sys
import time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

BASELINE = {
    "curvature_scaling": 1.0,
    "advection_scaling": 1.0, # shouldn't affect anything in this exp
    "propagation_scaling": 1.0,
    "num_iterations": 600,
    "max_rms_error": 0.0001
}

def _fmt_param(value: float, decimals: int = 2):
    sign = "n" if value < 0 else ""
    return f"{sign}{abs(value):.{decimals}f}".replace(".", "p")

def make_mask_name(curv, prop, adv=None):
    parts = [f"c{_fmt_param(curv)}", f"p{_fmt_param(prop)}"]

    if adv is not None:
        parts.append(f"a{_fmt_param(adv)}")

    return "_".join(parts)

def _get_bbox_with_padding(mask_np, padding_voxels):
    coords = np.argwhere(mask_np)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1

    pad = np.asarray(padding_voxels)
    mins = np.maximum(mins - pad, 0)
    maxs = np.minimum(maxs + pad, np.array(mask_np.shape))

    return mins, maxs

def _run_gac(initial_level_set, edge_potential, params):
    active_contour = sitk.GeodesicActiveContourLevelSetImageFilter()
    active_contour.SetCurvatureScaling(params["curvature_scaling"])
    active_contour.SetAdvectionScaling(params["advection_scaling"])
    active_contour.SetPropagationScaling(params["propagation_scaling"])
    active_contour.SetMaximumRMSError(params["max_rms_error"])
    active_contour.SetNumberOfIterations(params["num_iterations"])
    
    start = time.time()
    final_level_set = active_contour.Execute(initial_level_set, edge_potential)
    end = time.time()

    final_level_set_np = sitk.GetArrayFromImage(final_level_set)
    roi_binary_mask = (final_level_set_np < 0).astype(np.uint8)

    num_of_it = active_contour.GetElapsedIterations()
    rms_change = active_contour.GetRMSChange()

    data = {
        "num_of_it": int(num_of_it),
        "rms_change": float(rms_change),
        "time": f"{(end - start):.4f}"
    }

    return roi_binary_mask, data

def _process_patient(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    mask_sitk, mask = utils.scan_to_np_array(scan_path=os.path.join(data_dir, "final_mask_nip.seg.nrrd"), return_sitk=True)

    padding_mm = 5.0
    spacing_xyz = mask_sitk.GetSpacing()
    padding_voxels = [
        int(np.ceil(padding_mm / spacing_xyz[2])),
        int(np.ceil(padding_mm / spacing_xyz[1])),
        int(np.ceil(padding_mm / spacing_xyz[0]))
    ]
    mins, maxs = _get_bbox_with_padding(
        mask_np=mask, padding_voxels=padding_voxels
    )
    z_min, y_min, x_min = mins
    z_max, y_max, x_max = maxs

    roi_index = [int(x_min), int(y_min), int(z_min)]
    roi_size = [int(x_max - x_min), int(y_max - y_min), int(z_max - z_min)]

    mask_sitk_roi = sitk.RegionOfInterest(mask_sitk, roi_size, roi_index)

    signed_distance = sitk.SignedMaurerDistanceMap(
        mask_sitk_roi,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True
    )
    initial_level_set = sitk.Cast(signed_distance, sitk.sitkFloat32)

    edge_potential = initial_level_set * 0

    iterations = [10, 50, 100, 250, 500, 750, 1000, 2000, 5000]

    for iteration in iterations:
        curr_output_dir = os.path.join(output_dir, f"it{iteration}")
        os.makedirs(curr_output_dir, exist_ok=True)

        params = BASELINE.copy()
        params["num_iterations"] = iteration

        roi_binary_mask, _ = _run_gac(
            initial_level_set=initial_level_set,
            edge_potential=edge_potential,
            params=params
        )

        final_binary_mask = np.zeros_like(mask, np.uint8)
        final_binary_mask[z_min:z_max, y_min:y_max, x_min:x_max] = roi_binary_mask

        name = make_mask_name(
            curv=params["curvature_scaling"],
            prop=params["propagation_scaling"]
        )
        utils.save_data(
            data=final_binary_mask,
            ref_sitk=mask_sitk,
            output_dir=curr_output_dir,
            name=name,
            is_mask=True,
            color="0.1 0.45 0.8",
            segment_name=name
        )

def main():
    data_dir = os.path.join(ROOT_DIR, "pipeline_output")
    patient_id = "patient_0001"

    output_dir = "gac_exp_output_ep_0"

    _process_patient(
        data_dir=os.path.join(data_dir, patient_id),
        output_dir=os.path.join(output_dir, patient_id)
    )

if __name__ == "__main__":
    main()