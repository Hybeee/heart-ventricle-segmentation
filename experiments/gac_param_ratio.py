import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndimage

import json
import os, sys
import time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

BASELINE = {
    "curvature_scaling": 4.0,
    "advection_scaling": 1.0,
    "propagation_scaling": 0.0,
    "sigma": 2.0,
    "num_iterations": 5000,
    "max_rms_error": 0.0001,
}

BASELINE_RATIO_IT = {
    "curvature_scaling": 1.9802,
    "advection_scaling": 0.0198,
    "propagation_scaling": 0.0,
    "sigma": 2.0,
    "num_iterations": 5000,
    "max_rms_error": 0.00001,
}

def _build_param_sets():
    total_scale = 2.0
    ratios = np.array([0.01, 0.02, 0.05, 0.10, 0.20, 0.33, 0.50, 1.00, 1.50, 2.00, 3.00, 5.00, 10.00])

    advection_scaling = total_scale * ratios / (1.0 + ratios)
    curvature_scaling = total_scale / (1.0 + ratios)

    return {
        "ratio": ratios,
        "advection_scaling": advection_scaling,
        "curvature_scaling": curvature_scaling,
    }

def _build_it_param_set():
    return {
        "num_iterations": [100, 600, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 30000, 50000, 100000]
        # "num_iterations": [50000, 100000]
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

def _process_patient(patient_dir, output_dir, param_sets):
    sweep_data = []
    sd_out_path = os.path.join(output_dir, "sweep_data.json")

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

    for r, advec, curv in zip(param_sets["ratio"], param_sets["advection_scaling"], param_sets["curvature_scaling"]):
        print(f"advection_scaling={advec:.4f} | curvature_scaling={curv:.4f}")

        params = BASELINE.copy()
        params["advection_scaling"] = advec
        params["curvature_scaling"] = curv

        roi_binary_mask, data = _run_gac(
            signed_distance=signed_distance,
            initial_level_set=initial_level_set,
            params=params
        )

        final_binary_mask = np.zeros_like(mask, dtype=np.uint8)
        final_binary_mask[z_min:z_max, y_min:y_max, x_min:x_max] = roi_binary_mask
        name = f"ratio_{r}"
        utils.save_data(
            data=final_binary_mask,
            ref_sitk=orig_mask_sitk,
            output_dir=output_dir,
            name=name,
            is_mask=True,
            color="0.1 0.45 0.85",
            segment_name=name
        )

        run_data = {}
        run_data["advection_scaling"] = advec
        run_data["curvature_scaling"] = curv
        run_data.update(data)
        sweep_data.append(run_data)

    with open(sd_out_path, 'w') as f:
        json.dump(sweep_data, f, indent=2)

def _process_it_patient(patient_dir, output_dir, param_sets):
    sweep_data = []
    sd_out_path = os.path.join(output_dir, "sweep_data.json")

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

    for it in param_sets["num_iterations"]:
        print(f"num_iterations={it}")

        params = BASELINE_RATIO_IT.copy()
        params["num_iterations"] = it

        roi_binary_mask, data = _run_gac(
            signed_distance=signed_distance,
            initial_level_set=initial_level_set,
            params=params
        )

        final_binary_mask = np.zeros_like(mask, dtype=np.uint8)
        final_binary_mask[z_min:z_max, y_min:y_max, x_min:x_max] = roi_binary_mask
        name = f"it_{it:.2f}"
        utils.save_data(
            data=final_binary_mask,
            ref_sitk=orig_mask_sitk,
            output_dir=output_dir,
            name=name,
            is_mask=True,
            color="0.1 0.45 0.85",
            segment_name=name
        )

        run_data = {}
        run_data["it"] = it
        run_data.update(data)
        sweep_data.append(run_data)

    with open(sd_out_path, 'w') as f:
        json.dump(sweep_data, f, indent=2)

def main():
    data_dir = os.path.join(ROOT_DIR, "pipeline_output")
    output_dir = os.path.join(ROOT_DIR, "gac_p1_it_output")
    os.makedirs(output_dir, exist_ok=True)

    # process single patient
    patient_id = "patient_0001"
    patient_dir = os.path.join(data_dir, patient_id)
    patient_output_dir = os.path.join(output_dir, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)
    
    param_sets = _build_it_param_set()

    print("==============")
    print("PARAM SET")
    for k, v in param_sets.items():
        print(f"\t{k}: {v}")
    print("==============")

    # _process_patient(
    #     patient_dir=patient_dir,
    #     output_dir=patient_output_dir,
    #     param_sets=param_sets
    # )

    _process_it_patient(
        patient_dir=patient_dir,
        output_dir=patient_output_dir,
        param_sets=param_sets
    )

if __name__ == "__main__":
    main()