import numpy as np
from skimage.morphology import convex_hull_image
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import SimpleITK as sitk
import matplotlib.pyplot as plt

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

def _get_convex_hull(mask):
    coords = np.argwhere(mask)

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    cropped_mask = mask[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]

    cropped_hull = convex_hull_image(cropped_mask)
    convex_hull = np.zeros_like(mask)
    convex_hull[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1] = cropped_hull

    return convex_hull

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

def _get_relevant_points_mask(mask_b, ch_b):
    mask_b_points = np.argwhere(mask_b)
    ch_b_points = np.argwhere(ch_b)
    
    tree = KDTree(ch_b_points)
    distances, _ = tree.query(mask_b_points)

    threshold = np.percentile(distances, 80)
    print(f"Distance threshold: {threshold}")
    far_points = mask_b_points[distances > threshold]

    far_mask = np.zeros_like(mask_b)
    far_mask[tuple(far_points.T)] = 1
    
    far_mask_labeled, n_components = ndimage.label(far_mask)
    components, counts = np.unique(far_mask_labeled[far_mask_labeled > 0], return_counts=True)
    top3 = components[np.argsort(counts)[-3:]]
    far_mask = np.isin(far_mask_labeled, top3)

    # for comp in top3:
    #     comp_points = np.argwhere(far_mask_labeled == comp)
    #     zmin, ymin, xmin = comp_points.min(axis=0)
    #     zmax, ymax, xmax = comp_points.max(axis=0)

    #     radius = min((zmax-zmin)/2, (ymax-ymin)/2, (xmax-xmin)/2)

    #     print(f"Radius: {radius}")

    return far_mask

def make_ball(radius):
    r = int(radius)
    coords = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
    return sum(c**2 for c in coords) <= radius**2

def closing_downsampled(mask, radius, factor=2):
    small = ndimage.zoom(mask.astype(float), 1/factor, order=0) > 0.5
    ball = make_ball(max(1, int(radius / factor)))

    print(f"Running closing on mask shape: {small.shape} with ball shape: {ball.shape}")
    closed_small = ndimage.binary_closing(small, structure=ball)
    closed = ndimage.zoom(closed_small.astype(float), factor, order=1) > 0.5

    pad_width = [(0, max(0, mask.shape[i] - closed.shape[i])) for i in range(3)]
    closed = np.pad(closed, pad_width, mode='edge')
    closed = closed[tuple(slice(0, s) for s in mask.shape)]

    return closed

def run_for_patient(patient_dir):
    output_dir = os.path.join(patient_dir, "heart_muscle_data")
    os.makedirs(output_dir, exist_ok=True)

    mask_sitk, mask = utils.scan_to_np_array(scan_path=os.path.join(patient_dir, "final_mask_nip.seg.nrrd"), return_sitk=True)

    mask = _fill_internal_holes(mask=mask)
    boundary_mask = _get_mask_boundary(mask=mask)

    ch = _get_convex_hull(mask=mask)
    ch_b = _get_mask_boundary(mask=ch)

    far_mask = _get_relevant_points_mask(
        mask_b=boundary_mask,
        ch_b=ch_b
    )

    utils.save_data(
        data=far_mask,
        ref_sitk=mask_sitk,
        output_dir=output_dir,
        name="far_mask",
        is_mask=True,
        color="1.0 0.2 0.2",
        segment_name="far_mask"
    )

    utils.save_data(
        data=boundary_mask,
        ref_sitk=mask_sitk,
        output_dir=output_dir,
        name="boundary_mask",
        is_mask=True,
        color="0.2 0.8 1.0",
        segment_name="boundary_mask"
    )

    utils.save_data(
        data=ch_b,
        ref_sitk=mask_sitk,
        output_dir=output_dir,
        name="convex_hull_boundary",
        is_mask=True,
        color="0.2 1.0 0.4",
        segment_name="convex_hull_boundary"
    )

def main():

    dir = os.path.join(ROOT_DIR, "postproc_alg_vars_output_hm")
    for patient_id in os.listdir(dir):
        print(patient_id)
        run_for_patient(patient_dir=os.path.join(dir, patient_id))

    # patient_id = "patient_0008"
    # output_dir = os.path.join("streaking_viewer_output", patient_id)

    # ct = utils.scan_to_np_array(scan_path=os.path.join(output_dir, "ct.nii.gz"))
    # mask_sitk, mask = utils.scan_to_np_array(scan_path=os.path.join(output_dir, "final_mask_nip.seg.nrrd"), return_sitk=True)

    # mask = _fill_internal_holes(mask=mask)
    # boundary_mask = _get_mask_boundary(mask=mask)

    # chull_path = os.path.join(output_dir, "ch_b.seg.nrrd")
    # if os.path.exists(chull_path):
    #     ch_b = utils.scan_to_np_array(scan_path=chull_path)
    # else:
    #     ch = _get_convex_hull(mask=mask)
    #     ch_b = _get_mask_boundary(mask=ch)

    # far_mask = _get_relevant_points_mask(
    #     mask_b=boundary_mask,
    #     ch_b=ch_b
    # )
    
    # # ball = make_ball(13)
    # # closed_mask = ndimage.binary_closing(mask, structure=ball)
    # closed_mask = closing_downsampled(mask=mask, radius=13)
    # diff = closed_mask & ~mask
    # diff = ndimage.binary_opening(diff, structure=np.ones((3, 3, 3)))

    # diff_labeled, n_components = ndimage.label(diff)
    # components, counts = np.unique(diff_labeled[diff_labeled > 0], return_counts=True)
    # top2 = components[np.argsort(counts)[-2:]]
    # muscle_mask = np.isin(diff_labeled, top2)

    # print("Saving data")
    # utils.save_data(
    #     data=far_mask,
    #     ref_sitk=mask_sitk,
    #     output_dir=output_dir,
    #     name="far_mask",
    #     is_mask=True,
    #     color="1.0 0.2 0.2",
    #     segment_name="far_mask"
    # )

    # utils.save_data(
    #     data=boundary_mask,
    #     ref_sitk=mask_sitk,
    #     output_dir=output_dir,
    #     name="boundary_mask",
    #     is_mask=True,
    #     color="0.2 0.8 1.0",
    #     segment_name="boundary_mask"
    # )

    # utils.save_data(
    #     data=muscle_mask,
    #     ref_sitk=mask_sitk,
    #     output_dir=output_dir,
    #     name="muscle_mask",
    #     is_mask=True,
    #     color="0.2 1.0 0.4",
    #     segment_name="muscle_mask"
    # )

if __name__ == "__main__":
    main()