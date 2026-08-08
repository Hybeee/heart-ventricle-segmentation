import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

CONFIG = {
    "initial_level_set": "mask",
    "edge_potential": "10-eps"
}

def _get_mask_boundary(mask):
    mask_e = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3)))
    boundary = mask ^ mask_e
    points = np.argwhere(boundary)

    boundary_mask = np.zeros_like(mask)
    boundary_mask[tuple(points.T)] = 1

    return boundary_mask, points

def fill_internal_holes(mask):
    se = np.ones((3, 3))

    mask_closed = ndimage.binary_closing(mask, structure=se)
    mask_closed_eroded = ndimage.binary_erosion(mask_closed, structure=se)
    mask_filled = mask | mask_closed_eroded

    return mask_filled

def get_largest_component(mask):
    labels, num = ndimage.label(mask)

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    largest_label = np.argmax(sizes)
    return labels == largest_label

def plot(ct, mask, orig_mask, out_path):
    fig = plt.figure()
    plt.imshow(ct, cmap='gray')
    plt.imshow(mask, cmap='Blues', alpha=0.3)

    eroded_mask = binary_erosion(orig_mask).astype(orig_mask.dtype)
    boundary_mask = orig_mask - eroded_mask
    y_coords, x_coords = np.where(boundary_mask > 0)
    plt.scatter(x_coords, y_coords, s=2, c='red', marker='.')

    plt.axis('off')
    plt.savefig(out_path)
    plt.close(fig)

def _get_initial_level_set(mask, nnunet_mask, slice_spacing):
    mode = CONFIG["initial_level_set"]

    mask_sitk = sitk.GetImageFromArray(mask)
    mask_sitk.SetSpacing(slice_spacing)

    if mode == "mask":
        signed_distance = sitk.SignedMaurerDistanceMap(
            mask_sitk,
            insideIsPositive=False,
            squaredDistance=False,
            useImageSpacing=True
        )
        initial_level_set = sitk.Cast(signed_distance, sitk.sitkFloat32)

        return signed_distance, initial_level_set
    elif mode == "nnunet":
        mask_signed_distance = sitk.SignedMaurerDistanceMap(
            mask_sitk,
            insideIsPositive=False,
            squaredDistance=False,
            useImageSpacing=True
        )

        nnunet_sitk = sitk.GetImageFromArray(nnunet_mask)
        nnunet_sitk.SetSpacing(slice_spacing)
        nnunet_signed_distance = sitk.SignedMaurerDistanceMap(
            nnunet_sitk,
            insideIsPositive=False,
            squaredDistance=False,
            useImageSpacing=True
        )
        initial_level_set = sitk.Cast(nnunet_signed_distance, sitk.sitkFloat32)
        return mask_signed_distance, initial_level_set

def _get_edge_potential(ct, slice_spacing, sigma, initial_level_set, signed_distance):
    mode = CONFIG["edge_potential"]

    signed_distance_np = sitk.GetArrayFromImage(signed_distance)
    if mode == "10-eps":
        edge_potential_np = 1.0 - np.exp(-np.abs(signed_distance_np) / sigma)
    elif mode == "eps":    
        edge_potential_np = np.exp(-np.abs(signed_distance_np) / sigma)
    elif mode == "grad":    
        ct_sitk = sitk.GetImageFromArray(ct)
        ct_sitk.SetSpacing(slice_spacing)
        gradient_magnitude = sitk.GradientMagnitudeRecursiveGaussian(
            ct_sitk,
            sigma=sigma
        )
        grad_map_np = sitk.GetArrayFromImage(gradient_magnitude)
        edge_potential_np = 1.0 / (1.0 + grad_map_np)

    edge_potential = sitk.GetImageFromArray(edge_potential_np.astype(np.float32))
    edge_potential.CopyInformation(initial_level_set)
    return edge_potential

def _process_single_patient(data_dir, patient_id, cor_slice, case, out_dir):
    patient_dir = os.path.join(data_dir, patient_id)

    ct_sitk, ct = utils.scan_to_np_array(scan_path=os.path.join(patient_dir, "ct.nii.gz"), return_sitk=True)
    orig_mask_sitk, mask = utils.scan_to_np_array(scan_path=os.path.join(patient_dir, "final_mask_nip.seg.nrrd"), return_sitk=True)
    nnunet_mask = utils.scan_to_np_array(scan_path=os.path.join(patient_dir, "nnunet_mask.seg.nrrd"))

    ct = np.flipud(ct[:, cor_slice, :])

    mask = np.flipud(mask[:, cor_slice, :])
    mask = fill_internal_holes(mask) 
    mask = get_largest_component(mask)

    nnunet_mask = np.flipud(nnunet_mask[:, cor_slice, :])

    mask = mask.astype(np.uint8)
    orig_spacing = orig_mask_sitk.GetSpacing()
    slice_spacing = (orig_spacing[0], orig_spacing[2])

    signed_distance, initial_level_set = _get_initial_level_set(
        mask=mask,
        nnunet_mask=nnunet_mask
    )

    edge_potential = _get_edge_potential(
        ct=ct,
        slice_spacing=slice_spacing,
        sigma=case["sigma"],
        initial_level_set=initial_level_set,
        signed_distance=signed_distance
    )

    active_contour = sitk.GeodesicActiveContourLevelSetImageFilter()
    active_contour.SetCurvatureScaling(case["curvature_scaling"])
    active_contour.SetAdvectionScaling(case["advection_scaling"])  
    active_contour.SetPropagationScaling(case["propagation_scaling"])
    active_contour.SetMaximumRMSError(case["max_rms_error"])
    active_contour.SetNumberOfIterations(case["num_iterations"])

    final_level_set = active_contour.Execute(initial_level_set, edge_potential)

    final_level_set_np = sitk.GetArrayFromImage(final_level_set)
    final_binary_mask = (final_level_set_np < 0).astype(np.uint8)

    plot(ct, final_binary_mask, mask, os.path.join(out_dir, f"{case['name']}.png"))

def generate_test_cases(baseline, params):
    cases = [{"name": "baseline", **baseline}]
    for key, values in params.items():
        for v in values:
            if v == baseline[key]:
                continue
            case = dict(baseline)
            case[key] = v
            case["name"] = f"{key}={v}"
            cases.append(case)

    return cases

def run_tests(data_dir):
    baseline = {
        "curvature_scaling": 4.0,
        "advection_scaling": 1.0,
        "propagation_scaling": 0.3,
        "sigma": 8.0,
        "num_iterations": 600,
        "max_rms_error": 0.0001,
    }

    params = {
        "curvature_scaling":   [0.5, 1.0, 2.0, 4.0, 10.0, 20.0],
        "advection_scaling":   [0.0, 0.5, 1.0, 2.0, 5.0],
        "propagation_scaling": [0.0, 0.1, 0.3, 0.5, 1.0, 2.0],
        "sigma":               [1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 30.0], # [0.3, 0.5, 1.0, 1.5, 2.0, 3.0],
        "max_rms_error":       [0.01, 0.001, 0.0001, 0.00001],
        "num_iterations":      [100, 300, 600, 1000, 2000],
    }

    patients = ["patient_0001", "patient_0008", "patient_0019", "patient_0024"]
    slices = [197, 229, 207, 203]
    test_cases = generate_test_cases(baseline=baseline, params=params)

    for patient_id, cor_slice in zip(patients, slices):
        out_dir = os.path.join(ROOT_DIR, "snake_sweep_output_grad", patient_id)
        os.makedirs(out_dir, exist_ok=True)

        print(f"{patient_id}")
        for case in test_cases:
            print(f"\tTest case: {case['name']}")
            _process_single_patient(
                data_dir=data_dir,
                patient_id=patient_id,
                cor_slice=cor_slice,
                case=case,
                out_dir=out_dir
            )

def main():
    data_dir = os.path.join(ROOT_DIR, "pipeline_output")
    patient_id = "patient_0001"

    # _process_single_patient(data_dir, patient_id)
    run_tests(data_dir=data_dir)

if __name__ == "__main__":
    main()