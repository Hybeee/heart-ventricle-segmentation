import numpy as np
import scipy.ndimage as ndimage

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

def _get_mask_boundary(mask):
    mask_e = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3)))
    boundary = mask ^ mask_e
    points = np.argwhere(boundary)

    boundary_mask = np.zeros_like(mask)
    boundary_mask[tuple(points.T)] = 1

    return boundary_mask

def _fill_internal_holes(mask):
    se_26_connectivity = np.ones((3, 3, 3))
    connected_components, _ = ndimage.label(~mask, se_26_connectivity)

    components, component_counts = np.unique(connected_components, return_counts=True)

    max_component = components[np.argmax(component_counts)]

    filled_mask = mask | (connected_components != max_component)

    return filled_mask

def _get_mask_center(mask):
    zs, ys, xs = np.nonzero(mask)
    return (int(zs.mean()), int(ys.mean()), int(xs.mean()))

def _to_spherical(points, center, spacing):
    d = (points - center) * spacing # * spacing => phyiscal coordinates
    r = np.linalg.norm(d, axis=1)
    theta = np.arccos(np.clip(d[:, 0] / r, -1, 1))
    phi = np.arctan2(d[:, 1], d[:, 2])

    sph_points = np.stack((r, theta, phi), axis=-1)

    return sph_points

def process_patient(dir, patient_id):
    patient_dir = os.path.join(dir, patient_id) 

    spacing, ct = utils.scan_to_np_array(scan_path=os.path.join(patient_dir, "ct.nii.gz"), return_spacing=True)
    mask_sitk, mask = utils.scan_to_np_array(scan_path=os.path.join(patient_dir, "final_mask_nip.seg.nrrd"), return_sitk=True)

    mask = _fill_internal_holes(mask=mask)
    mask_b = _get_mask_boundary(mask=mask)
    boundary_points = np.column_stack(np.where(mask_b == 1))

    center = _get_mask_center(mask=mask)

    print("Converting to spherical...")
    sph_points = _to_spherical(boundary_points, center, spacing)
    print(f"Spherical points shape: {sph_points.shape}")
    print(f"Example points: {sph_points[23]}")

def main():
    dir = os.path.join(ROOT_DIR, "streaking_viewer_output")
    patient_id = "patient_0001"
    process_patient(
        dir=dir,
        patient_id=patient_id
    )

if __name__ == "__main__":
    main()