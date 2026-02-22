import numpy as np
import cv2
from scipy.ndimage import convolve, gaussian_filter
import matplotlib.pyplot as plt
from skimage import measure
import SimpleITK as sitk
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator

import os

def scan_to_np_array(scan_path, return_sitk=False, return_spacing=False):
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

    if return_sitk:
        return orig_scan, scan
    elif return_spacing:
        return spacing, scan
    else:
        return scan
    
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