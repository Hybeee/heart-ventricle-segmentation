import SimpleITK as sitk
import numpy as np

import os
import sys
import time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

PRINTED = False

BASELINE = {
    "curvature_scaling": 1.0,
    "advection_scaling": 1.0, # shouldn't affect anything in this exp
    "propagation_scaling": 0.0,
    "num_iterations": 600,
    "max_rms_error": 0.0001
}

def _fmt_param(value: float, decimals: int = 4):
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
    global PRINTED

    active_contour = sitk.GeodesicActiveContourLevelSetImageFilter()
    active_contour.SetCurvatureScaling(params["curvature_scaling"])
    active_contour.SetAdvectionScaling(params["advection_scaling"])
    active_contour.SetPropagationScaling(params["propagation_scaling"])
    active_contour.SetMaximumRMSError(params["max_rms_error"])
    active_contour.SetNumberOfIterations(params["num_iterations"])

    if not PRINTED:
        for k, v in params.items():
            if k == "num_iterations" or k == "max_rms_error":
                continue

            print(f"\t{k}={v}")

    PRINTED = True

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

def _process_patient(data_dir, output_dir, params, iterations=None):
    global PRINTED
    PRINTED = False

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
    signed_distance_np = sitk.GetArrayFromImage(signed_distance)

    initial_level_set = sitk.Cast(signed_distance, sitk.sitkFloat32)

    # edge_potential = initial_level_set * 0 + 1
    
    # noise
    # noise_amplitude = 0.05
    # rng = np.random.default_rng(seed=42)
    # shape = sitk.GetArrayFromImage(initial_level_set).shape
    # noise_np = rng.normal(loc=1.0, scale=noise_amplitude, size=shape).astype(np.float32)
    # noise_np = np.clip(noise_np, 1e-3, None)
    # edge_potential = sitk.GetImageFromArray(noise_np)
    # edge_potential.CopyInformation(initial_level_set)

    # normal
    floor = 0.1
    edge_potential_np = floor + (1 - floor) * (1.0 - np.exp(-(signed_distance_np ** 2) / (2 * 2.0 ** 2)))
    edge_potential = sitk.GetImageFromArray(edge_potential_np.astype(np.float32))
    edge_potential.CopyInformation(initial_level_set)

    if iterations is None:
        iterations = [10, 50, 100, 250, 500, 750, 1000, 2000, 5000]

    for iteration in iterations:
        curr_output_dir = os.path.join(output_dir, f"it{iteration}")
        os.makedirs(curr_output_dir, exist_ok=True)

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
            prop=params["propagation_scaling"],
            adv=params["advection_scaling"]
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

def _run_sweep(data_dir, output_dir, params=None):
    iterations = [10, 20, 30, 40, 50, 60, 70, 80, 90,
                100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500,
                550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

    total_scale = 2

    # ratios = np.round(np.geomspace(0.01, 10, num=25), 4)

    low = np.geomspace(0.01, 1.0, num=6)
    high = np.geomspace(1.0, 100.0, num=20)[1:]
    ratios = np.concatenate([low, high])

    print(ratios)
    
    # curvature_scaling = total_scale * ratios / (1.0 + ratios)
    # propagation_scaling = total_scale / (1.0 + ratios)
    curvature_scaling = [params["curvature_scaling"]]
    propagation_scaling = [params["propagation_scaling"]] 

    for i, (curv, prop) in enumerate(zip(curvature_scaling, propagation_scaling)):
        print(f"Processing {i+1}/{len(curvature_scaling)} iteration; curv={curv} prop={prop}")

        params = params.copy() if params is not None else BASELINE.copy()
        params["curvature_scaling"] = curv
        params["propagation_scaling"] = prop

        _process_patient(
            data_dir=data_dir,
            output_dir=output_dir,
            params=params,
            iterations=iterations
        )

def _run_advec_sweep(data_dir, output_dir):
    iterations = [10, 20, 30, 40, 50, 60, 70, 80, 90,
            100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
            225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500,
            550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

    low = np.geomspace(0.01, 1.0, num=6)
    high = np.geomspace(1.0, 100.0, num=20)[1:]
    ratios = np.concatenate([low, high])

    base_params = {
        "curvature_scaling": 1.5413,
        "advection_scaling": None,
        "propagation_scaling": 0.4587,
        "num_iterations": 600,
        "max_rms_error": 0.0001
    }

    advection_scaling = base_params["curvature_scaling"] / ratios
    print(advection_scaling)

    for i, adv in enumerate(advection_scaling):
        ratio = base_params["curvature_scaling"] / adv
        print(f"Processing {i+1}/{len(advection_scaling)} iteration; adv={adv} ratio={ratio}")

        params = base_params.copy()
        params["advection_scaling"] = adv

        _process_patient(
            data_dir=data_dir,
            output_dir=output_dir,
            params=params,
            iterations=iterations
        )

def _run_prop_sweep(data_dir, output_dir):
    iterations = [10, 20, 30, 40, 50, 60, 70, 80, 90,
            100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
            225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500,
            550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

    base_params = {
        "curvature_scaling": 1.5413,
        "advection_scaling": 0.5,
        "propagation_scaling": 0.4587,
        "num_iterations": 600,
        "max_rms_error": 0.0001
    }

    propagation_scaling = [
        0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0
    ]
    print(propagation_scaling)

    for i, prop in enumerate(propagation_scaling):
        print(f"Processing {i+1}/{len(propagation_scaling)} iteration; prop={prop}")

        params = base_params.copy()
        params["propagation_scaling"] = prop

        _process_patient(
            data_dir=data_dir,
            output_dir=output_dir,
            params=params,
            iterations=iterations
        )

def _run_advcurv_sweep(data_dir, output_dir, params=None):
    iterations = [10, 20, 30, 40, 50, 60, 70, 80, 90,
            100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
            225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500,
            550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

    total_scale = 2

    # ratios = np.round(np.geomspace(0.01, 10, num=25), 4)

    low = np.geomspace(0.01, 1.0, num=6)
    high = np.geomspace(1.0, 100.0, num=20)[1:]
    ratios = np.concatenate([low, high])

    print(ratios)
    
    curvature_scaling = total_scale * ratios / (1.0 + ratios)
    advection_scaling = total_scale / (1.0 + ratios)

    for i, (curv, adv) in enumerate(zip(curvature_scaling, advection_scaling)):
        print(f"Processing {i+1}/{len(curvature_scaling)} iteration; curv={curv} adv={adv}")

        params = params.copy() if params is not None else BASELINE.copy()
        params["curvature_scaling"] = curv
        params["advection_scaling"] = adv

        _process_patient(
            data_dir=data_dir,
            output_dir=output_dir,
            params=params,
            iterations=iterations
        )

def main():
    data_dir = os.path.join(ROOT_DIR, "pipeline_output")
    patient_id = "patient_0005"

    output_dir = os.path.join(ROOT_DIR, "gac_exp_sweep_output_ratio_test")

    params = {
        "curvature_scaling": 1.00,
        "advection_scaling": 1.00,
        "propagation_scaling": 0.2976,
        "num_iterations": 600,
        "max_rms_error": 0.0001
    }

    iterations = [10, 20, 30, 40, 50, 60, 70, 80, 90,
            100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
            225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500,
            550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

    _process_patient(
        data_dir=os.path.join(data_dir, patient_id),
        output_dir=os.path.join(output_dir, patient_id),
        params=params,
        iterations=iterations
    )

    # _run_sweep(
    #     data_dir=os.path.join(data_dir, patient_id),
    #     output_dir=os.path.join(output_dir, patient_id),
    #     params=params
    # )

    # _run_advcurv_sweep(
    #     data_dir=os.path.join(data_dir, patient_id),
    #     output_dir=os.path.join(output_dir, patient_id),
    #     params=params
    # )

    # _run_advec_sweep(
    #     data_dir=os.path.join(data_dir, patient_id),
    #     output_dir=os.path.join(output_dir, patient_id)
    # )

    # _run_prop_sweep(
    #     data_dir=os.path.join(data_dir, patient_id),
    #     output_dir=os.path.join(output_dir, patient_id)
    # )

if __name__ == "__main__":
    main()