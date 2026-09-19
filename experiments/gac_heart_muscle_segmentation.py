import SimpleITK as sitk
import numpy as np

import os
import sys
import time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

GAC_PARAMS = {
    "curvature_scaling": 1.00,
    "advection_scaling": 1.00,
    "propagation_scaling": 0.2976,
    "num_iterations": 1000,
    "max_rms_error": 0.0001
}

GAC_ITERATIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90,
        100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
        225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500,
        550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

def _fmt_param(value: float, decimals: int = 4):
    sign = "n" if value < 0 else ""
    return f"{sign}{abs(value):.{decimals}f}".replace(".", "p")

def _make_mask_name(curv, prop, adv=None):
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

def _create_mask_hull(mask_sitk, mask, save_mask_history, out_dir):
    params = GAC_PARAMS.copy()
    iterations = GAC_ITERATIONS.copy()

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
    signed_distance_np = sitk.GetArrayFromImage(signed_distance)

    initial_level_set = sitk.Cast(signed_distance, sitk.sitkFloat32)

    floor = 0.1
    edge_potential_np = floor + (1 - floor) * (1.0 - np.exp(-(signed_distance_np ** 2) / (2 * 2.0 ** 2)))
    edge_potential = sitk.GetImageFromArray(edge_potential_np.astype(np.float32))
    edge_potential.CopyInformation(initial_level_set)

    if not save_mask_history:
        active_contour = sitk.GeodesicActiveContourLevelSetImageFilter()
        active_contour.SetCurvatureScaling(params["curvature_scaling"])
        active_contour.SetAdvectionScaling(params["advection_scaling"])
        active_contour.SetPropagationScaling(params["propagation_scaling"])
        active_contour.SetMaximumRMSError(params["max_rms_error"])
        active_contour.SetNumberOfIterations(params["num_iterations"])
        final_level_set = active_contour.Execute(initial_level_set, edge_potential)
    else:
        elapsed_iter = 0
        current_level_set = initial_level_set

        for iteration in iterations:
            curr_output_dir = os.path.join(out_dir, f"it{iteration}")
            os.makedirs(curr_output_dir, exist_ok=True)

            current_iter = iteration - elapsed_iter
            active_contour = sitk.GeodesicActiveContourLevelSetImageFilter()
            active_contour.SetCurvatureScaling(params["curvature_scaling"])
            active_contour.SetAdvectionScaling(params["advection_scaling"])
            active_contour.SetPropagationScaling(params["propagation_scaling"])
            active_contour.SetMaximumRMSError(params["max_rms_error"])
            active_contour.SetNumberOfIterations(current_iter)

            current_level_set = active_contour.Execute(current_level_set, edge_potential)
            elapsed_iter += active_contour.GetElapsedIterations()

            current_level_set_np = sitk.GetArrayFromImage(current_level_set)
            roi_binary_mask = (current_level_set_np < 0).astype(np.uint8)
            current_binary_mask = np.zeros_like(mask, np.uint8)
            current_binary_mask[z_min:z_max, y_min:y_max, x_min:x_max] = roi_binary_mask
            name = _make_mask_name(
                curv=params["curvature_scaling"],
                prop=params["propagation_scaling"],
                adv=params["advection_scaling"]
            )
            utils.save_data(
                data=current_binary_mask,
                ref_sitk=mask_sitk,
                output_dir=curr_output_dir,
                name=name,
                is_mask=True,
                color="0.1 0.45 0.8",
                segment_name=name
            )

            if active_contour.GetElapsedIterations() < current_iter:
                print(f"Hull calculation stopped after {elapsed_iter} iterations")
                break

        final_level_set = current_level_set

    final_level_set_np = sitk.GetArrayFromImage(final_level_set)
    roi_binary_mask = (final_level_set_np < 0).astype(np.uint8)
    final_binary_mask = np.zeros_like(mask, np.uint8)
    final_binary_mask[z_min:z_max, y_min:y_max, x_min:x_max] = roi_binary_mask
    utils.save_data(
        data=final_binary_mask,
        ref_sitk=mask_sitk,
        output_dir=out_dir,
        name="mask_hull",
        is_mask=True,
        color="0.1 0.45 0.8",
        segment_name="mask_hull"
    )

def _process_patient(data_dir, out_dir, save_mask_history=True):
    os.makedirs(out_dir, exist_ok=True)

    mask_sitk, mask = utils.scan_to_np_array(scan_path=os.path.join(data_dir, "final_mask_nip.seg.nrrd"), return_sitk=True)

    start = time.time()
    _create_mask_hull(mask_sitk, mask, save_mask_history, out_dir)
    end = time.time()

    print(f"\tHull created in {(end - start):.4f}s")

def main():
    data_dir = os.path.join(ROOT_DIR, "pipeline_output")
    patient_id = "patient_0001"

    out_dir = os.path.join(ROOT_DIR, "heart_muscle_segmentation_output")
    os.makedirs(out_dir, exist_ok=True)

    for patient_id in sorted(os.listdir(data_dir)):
        print(f"Processing {patient_id}...")

        _process_patient(
            data_dir=os.path.join(data_dir, patient_id),
            out_dir=os.path.join(out_dir, patient_id)
        )

if __name__ == "__main__":
    main()