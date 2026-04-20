import numpy as np
import cv2
from scipy.ndimage import convolve, gaussian_filter
from skimage.morphology import disk
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from skimage import measure
import SimpleITK as sitk
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

from postprocessor import ValleyData

import os

def scan_to_np_array(scan_path, return_all=False, return_sitk=False, return_spacing=False):
    """
    Returns the read scan in the following shape: (z, y, x).
    """
    orig_scan = sitk.ReadImage(scan_path) # shape: z, y, x
    # print(f"scan spacing: {orig_scan.GetSpacing()}")
    direction = orig_scan.GetDirection()
    if direction != (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0):
        orig_scan = sitk.DICOMOrient(orig_scan, "LPS")
    spacing = orig_scan.GetSpacing()
    spacing = spacing[::-1]
    scan = sitk.GetArrayFromImage(orig_scan)

    if return_all:
        return spacing, orig_scan, scan
    elif return_sitk:
        return orig_scan, scan
    elif return_spacing:
        return spacing, scan
    else:
        return scan

def save_data(data, ref_sitk, output_dir, name):
    img = sitk.GetImageFromArray(data)

    img.SetSpacing(ref_sitk.GetSpacing())
    img.SetDirection(ref_sitk.GetDirection())
    img.SetOrigin(ref_sitk.GetOrigin())

    sitk.WriteImage(img, os.path.join(output_dir, f"{name}.nii.gz"))

def create_and_save_lung_mask(patient_data, output_dir):
    input_img = nib.load(patient_data["ct_path"])
    output_img = totalsegmentator(input=input_img, task="total", verbose=False)

    seg_data = output_img.get_fdata()

    class_indices = [10, 11] # lung mask indices
    mask = np.isin(seg_data, class_indices)
    seg_data = mask.astype(np.uint8)

    output_path = os.path.join(output_dir, "lung_mask.nii.gz")
    result_nifti = nib.Nifti1Image(seg_data, output_img.affine)
    nib.save(result_nifti, output_path)

    patient_data["lung_mask_path"] = output_path

    print(f"Lung segmentation saved to: {output_path}")
    
    return patient_data

def calculate_slice_center(slice):
    ys, xs = np.nonzero(slice)
    return (int(ys.mean()), int(xs.mean()))

class PolarConverter():

    def __init__(self, dsize, maxRadius, center, flags):
        self.dsize = dsize
        self.maxRadius = maxRadius
        self.center = center
        self.flags = flags

    def cv2WarpPolar(self, image):
        return cv2.warpPolar(image,
                             dsize=self.dsize,
                             maxRadius=self.maxRadius,
                             center=self.center,
                             flags=self.flags).T
    
def get_mask_boundary(mask):
    mask_boundary = []
    contours = measure.find_contours(mask, level=0.5)
    for contour in contours:
        mask_boundary.extend(contour)
    
    return np.array(mask_boundary)


def get_directional_grad(ct, center, sigma, polarConverter: PolarConverter,
                         need_gaus=True, deg=1):
    if need_gaus:
        ct_gaus = gaussian_filter(ct, sigma=sigma)
    else:
        ct_gaus = ct
    x_mx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    Gx_ct = convolve(ct_gaus, np.flip(x_mx), mode="nearest")

    y_mx = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    Gy_ct = convolve(ct_gaus, np.flip(y_mx), mode="nearest")

    dir_deriv_ct = np.empty(shape=ct.shape, dtype=np.float32)

    Y, X = np.indices(ct.shape)

    dy = Y - center[0]
    dx = X - center[1]

    norm = np.sqrt(dy ** 2 + dx ** 2)
    norm[norm==0] = 1.0

    vy = dy / norm
    vx = dx / norm

    dir_deriv_ct = (vy * Gy_ct + vx * Gx_ct) * (sigma ** deg)

    polar_dir_grad_ct = polarConverter.cv2WarpPolar(image=dir_deriv_ct.astype(np.float32))

    return dir_deriv_ct, polar_dir_grad_ct

def get_scaled_sigma(curr_spacing, ref_pixel_sigma, ref_spacing):
    ref_spacing = np.asarray(ref_spacing[1:], dtype=float) # ref_spacing.shape = z, y, x
    curr_spacing = np.asarray(curr_spacing[1:], dtype=float)

    ref_physical_sigma = ref_pixel_sigma * ref_spacing

    scaled_pixel_sigma = ref_physical_sigma / curr_spacing

    return scaled_pixel_sigma

def get_polar_boundary_points(polar_mask, theta_step_size=15, only_last=True, append_default=True):
    points = []

    curr_theta = 0
    for curr_theta in range(0, polar_mask.shape[1] - (theta_step_size - 1), theta_step_size):
        curr_radius = 0
        curr_point = None
        for curr_radius in range(polar_mask.shape[0] - 1):
            prev_point = polar_mask[curr_radius, curr_theta]
            next_point = polar_mask[curr_radius + 1, curr_theta]

            if prev_point == 1 and next_point == 0:
                if only_last:
                    curr_point = np.array([curr_radius, curr_theta])
                else:
                    points.append(np.array([curr_radius, curr_theta]))
        
        if only_last:
            if curr_point is not None:
                points.append(curr_point)
            elif append_default:
                points.append(np.array([-1, curr_theta]))

    return np.array(points)

def clip_polar_gradients(polar_dir_grad_ct, grad_clip=2200):
        grad_image = np.nan_to_num(polar_dir_grad_ct, nan=0.0, posinf=grad_clip, neginf=-grad_clip)

        grad_image = np.clip(grad_image, -grad_clip, grad_clip)

        grad_image[grad_image.shape[0] - 1, 0] = -grad_clip
        grad_image[grad_image.shape[0] - 1, grad_image.shape[1] - 1] = grad_clip

        return grad_image

def plot_valley_data(valley_data: ValleyData, masks: list, mask_labels: list, theta: int,
                     save_dir: str, save_name: str, save: bool=True):
    plt.plot(valley_data.fo_deriv[:, theta][:100])
    plt.xlabel("Radii")
    plt.ylabel("Gradient value")
    plt.grid(True)

    curr_valleys_rs = valley_data.valley_positions[theta]
    curr_valleys_rs = curr_valleys_rs[curr_valleys_rs < 100]
    curr_valleys_values = valley_data.fo_deriv[curr_valleys_rs, theta]
    valley_color = np.random.rand(3,)

    plt.scatter(
        curr_valleys_rs,
        curr_valleys_values,
        s=5,
        color=valley_color,
        label='Valley'
    )

    for i, mask in enumerate(masks):
        mask_radius = mask[theta][0]
        mask_value = valley_data.fo_deriv[mask_radius, theta]

        random_color = np.random.rand(3,)

        plt.scatter(
            mask_radius,
            mask_value,
            s=5,
            color=random_color,
            label=mask_labels[i]
        )

    plt.legend()
    if save:
        plt.savefig(os.path.join(save_dir, f"{save_name}.png"))
        plt.close()
    else:
        plt.show()

def get_3d_mask(ct, best_threshold, ventricle_mask):
    assert ct.shape == ventricle_mask.shape and len(ct.shape) == 3

    best_threshold = float(best_threshold)

    mask = np.zeros_like(ct)
    mask[ct > best_threshold] = 1
    mask[ventricle_mask == 0] = 0

    return mask

def remove_segmentation_leakage(arc_mask, pixel_spacing):
        # fill small holes in the mask (the holes are created by the thresholding step)
        se_8_connectivity = np.zeros((3, 3, 3))
        se_8_connectivity[1] = 1
        connected_components, _ = ndimage.label(~arc_mask, se_8_connectivity)
        unique_values, values_counts = np.unique(connected_components, return_counts=True)
        big_hole_indices = unique_values[values_counts > ((2.5/pixel_spacing[1])**2)]
        arc_mask_imfilled = ~np.isin(connected_components, big_hole_indices) | arc_mask
        # calculate dt on the imfilled (only small holes are filled) mask
        arc_mask_dt = ndimage.distance_transform_edt(arc_mask_imfilled, sampling=[1000, 1, 1])  # "2D" dt
        # dilate dt values and compare it to the dt values to find seeds for the main part of the aorta
        diam_small = round(4/pixel_spacing[1])
        diam_small = diam_small - diam_small % 2 + 1

        radius = diam_small // 2
        structure = np.zeros((1, diam_small, diam_small), dtype=bool)

        yy, xx = np.ogrid[:diam_small, :diam_small]
        center = radius

        structure[0] = (yy - center) ** 2 + (xx - center) ** 2 <= radius ** 2

        dilated_dt = ndimage.grey_dilation(
            arc_mask_dt,
            footprint=structure
        )

        dt_max_values = dilated_dt.max(axis=(1, 2))

        dt_seed_mask = (
             np.isclose(dilated_dt, arc_mask_dt, rtol=0, atol=0.5)
             & (arc_mask_dt > 0)
             & (arc_mask_dt > dt_max_values[:, None, None] * 0.5)
        )

        # find the number of erosions that is needed to split from the main part of the aorta (iterative algo)
        # init:  marching_state[x] = dt[x], if x is a seed pont, else 0
        # marching_state[x]=max(marching_state[x],min(dt(x),max(marching_state[Neighbour(x)])
        seed_points = np.stack(dt_seed_mask.nonzero()).T
        neighbors_d = np.array([[0, -1, -1], [0, -1, 0], [0, -1, 1],
                                [0, 0, -1], [0, 0, 1],
                                [0, 1, -1], [0, 1, 0], [0, 1, 1]])
        new_points = seed_points
        marching_state = np.zeros_like(arc_mask_dt)
        marching_state[tuple(new_points.T)] = arc_mask_dt[tuple(new_points.T)]
        new_points_neighbors = new_points[:, None, :] + neighbors_d[None, ...]
        new_points = new_points_neighbors.reshape((-1, 3))
        while len(new_points) > 0:
            # print(len(new_points))
            new_points = np.unique(new_points, axis=0)
            neighbors = new_points[:, None, :] + neighbors_d[None, ...]
            current_seed_vals = marching_state[tuple(new_points.T)]
            neighbour_vals = marching_state[tuple(np.moveaxis(neighbors, 2, 0))]
            current_seed_dt_vals = arc_mask_dt[tuple(new_points.T)]
            marching_state[tuple(new_points.T)] = np.maximum(current_seed_vals,
                                                             np.minimum(current_seed_dt_vals,
                                                                        neighbour_vals.max(axis=-1)))
            changed_points = marching_state[tuple(new_points.T)] != current_seed_vals
            new_points_changed = new_points[changed_points]
            changed_points_neighbors = new_points_changed[:, None, :] + neighbors_d[None, ...]
            new_points = changed_points_neighbors.reshape((-1, 3))
        # select homogenous islands from the marching state values --> indicates leaks
        se = np.ones((1, 3, 3))
        march_erosion = ndimage.grey_erosion(
             marching_state,
             footprint=se
        )
        island_mask = (np.isclose(marching_state, march_erosion, rtol=0, atol=0.5)
                       & (marching_state > 0))
        # remove these islands
        max_connection_half_width = round(2.22 / pixel_spacing[1])
        n_erosions_needed = np.max((island_mask & (marching_state < max_connection_half_width))
                                   * marching_state, axis=(1, 2))
        reconstruction_base = marching_state > n_erosions_needed[:, None, None]
        # fix the parts removed from the main part by dilation
        arc_mask_reconstructed = np.array([ndimage.binary_dilation(base_mask, disk(round(d)), mask=mask)
                                           if d > 0 else base_mask
                                           for base_mask, d, mask in zip(reconstruction_base,
                                                                         n_erosions_needed,
                                                                         arc_mask_imfilled)])
        return arc_mask_reconstructed

def calculate_mask_metrics(prediction, ground_truth):
    assert prediction.shape == ground_truth.shape

    prediction = np.asarray(prediction).astype(bool)
    ground_truth = np.asarray(ground_truth).astype(bool)

    intersection = np.logical_and(prediction, ground_truth).sum()
    union = np.logical_or(prediction, ground_truth).sum()

    iou = intersection / union if union > 0 else 1.0
    dice = (2.0 * intersection) / (prediction.sum() + ground_truth.sum()) if (prediction.sum() + ground_truth.sum()) > 0 else 1.0

    result = {
        "iou": iou,
        "dice_score": dice
    }

    return result

def get_weighted_median(weights, values):
    values = np.array(values)
    weights = np.array(weights)

    sort_indices = np.argsort(values)

    values_sorted = values[sort_indices]
    weights_sorted = weights[sort_indices]

    cumsum = weights_sorted.cumsum()

    cutoff = weights_sorted.sum() / 2.

    return values_sorted[cumsum >= cutoff][0]

def main():
    import os
    polar_doc_mask = np.load(os.path.join("szaniszlo", "code", "output",
                                      "polar_outputs_unified_sigma",
                                      "patient_0001", "120", "data_np",
                                      "polar_doc_mask.npy"))
    
    doc_points = get_polar_boundary_points(polar_doc_mask, theta_step_size=15)
    np.save(os.path.join("szaniszlo", "code", "output",
                                      "polar_outputs_unified_sigma",
                                      "patient_0001", "120", "data_np",
                                      "polar_doc_mask_boundary.npy"), doc_points)
    
    polar_nnunet_mask = np.load(os.path.join("szaniszlo", "code", "output",
                                      "polar_outputs_unified_sigma",
                                      "patient_0001", "120", "data_np",
                                      "polar_nnunet_mask.npy"))
    
    nnunet_points = get_polar_boundary_points(polar_nnunet_mask, theta_step_size=15)
    np.save(os.path.join("szaniszlo", "code", "output",
                                      "polar_outputs_unified_sigma",
                                      "patient_0001", "120", "data_np",
                                      "polar_nnunet_mask_boundary.npy"), nnunet_points)
if __name__ == "__main__":
    main()