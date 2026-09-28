import numpy as np
from scipy.ndimage import convolve, gaussian_filter, map_coordinates, binary_erosion

import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

def sobel_3d(axis):
    deriv = np.array([-1, 0, 1])
    smooth = np.array([1, 2, 1])
    
    vecs = [smooth, smooth, smooth]
    vecs[axis] = deriv
    kernel = (vecs[0].reshape(-1, 1, 1) *
              vecs[1].reshape(1, -1, 1) *
              vecs[2].reshape(1, 1, -1))

    return kernel

def get_3d_dir_derivs(ct, center, sigma=2.0):
    ct_gaus = gaussian_filter(ct, sigma=2.0)

    z_mz = sobel_3d(axis=0)
    y_my = sobel_3d(axis=1)
    x_mx = sobel_3d(axis=2)

    Gz_ct = convolve(ct_gaus, np.flip(z_mz), mode="nearest")
    Gy_ct = convolve(ct_gaus, np.flip(y_my), mode="nearest")
    Gx_ct = convolve(ct_gaus, np.flip(x_mx), mode="nearest")

    dir_deriv_ct = np.empty(shape=ct.shape, dtype=np.float32)

    Z, Y, X = np.indices(ct.shape)

    dz = Z - center[0]
    dy = Y - center[1]
    dx = X - center[2]

    norm = np.sqrt(dz ** 2 + dy ** 2 + dx ** 2)
    norm[norm==0] = 1.0

    vz = dz / norm
    vy = dy / norm
    vx = dx / norm

    dir_deriv_ct = (vz * Gz_ct + vy * Gy_ct + vx * Gx_ct) * sigma

    return dir_deriv_ct

def _sample_at(volume, points):
    return map_coordinates(volume, points.T, order=1, mode='nearest')

def find_edges_along_rays(dir_deriv_ct, points, center, serach_radius=5.0, step=0.5):
    N = points.shape[0]

    d = points - center
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    directions = d / norm

    ts = np.arange(-serach_radius, serach_radius + step, step)
    T = ts.shape[0]

    candidates = points[:, None, :] + ts[None, :, None] * directions[:, None, :]
    candidate_points = candidates.reshape(-1, 3)

    point_vals = _sample_at(dir_deriv_ct, candidate_points)
    vals = point_vals.reshape(N, T)

    best_idx = np.argmin(vals, axis=1)
    best_vals = vals[np.arange(N), best_idx]
    best_points = candidates[np.arange(N), best_idx, :]

    return best_points, best_vals

def _get_boundary_points(mask):
    eroded = binary_erosion(mask)
    boundary = mask & ~eroded
    return np.argwhere(boundary).astype(np.float64)

def calculate_bde(mask, dir_deriv_ct, spacing, center):
    boundary_points = _get_boundary_points(mask)

    edge_pts, vals = find_edges_along_rays(
        dir_deriv_ct=dir_deriv_ct,
        points=boundary_points,
        center=center
    )

    diffs_mm = (edge_pts - boundary_points) * spacing
    dists = np.linalg.norm(diffs_mm, axis=1)

    neg = vals < 0
    pos = ~neg
    d_max = 5.0 * np.mean(spacing)

    w = np.empty_like(vals, dtype=np.float64)
    d = np.empty_like(vals, dtype=np.float64)

    w[neg] = -vals[neg]
    d[neg] = dists[neg]

    w_ref = w[neg].mean() if neg.any() else 1.0
    w[pos] = w_ref * (1.0 + vals[pos] / (np.abs(vals[neg]).mean() if neg.any() else 1.0))
    d[pos] = d_max

    bde = np.sum(w * d) / np.sum(w) if neg.any() else np.nan

    return bde

def main():
    data_root_dir = os.path.join(ROOT_DIR, "pipeline_output")

    for patient_id in sorted(os.listdir(data_root_dir)):
        print(patient_id)
        data_dir = os.path.join(data_root_dir, patient_id)

        spacing, ct = utils.scan_to_np_array(scan_path=os.path.join(data_dir, "ct.nii.gz"), return_spacing=True)
        final_mask = utils.scan_to_np_array(scan_path=os.path.join(data_dir, "final_mask_nip.seg.nrrd"))
        doc_mask = utils.scan_to_np_array(scan_path=os.path.join(data_dir, "doc_mask.seg.nrrd"))
        nnunet_mask = utils.scan_to_np_array(scan_path=os.path.join(data_dir, "nnunet_mask.seg.nrrd"))
        center = np.argwhere(final_mask > 0).mean(axis=0)
        
        dir_deriv_ct = get_3d_dir_derivs(
            ct=ct,
            center=center
        )

        fm = calculate_bde(mask=final_mask, dir_deriv_ct=dir_deriv_ct, spacing=spacing, center=center)
        doc = calculate_bde(mask=doc_mask, dir_deriv_ct=dir_deriv_ct, spacing=spacing, center=center)
        nnunet = calculate_bde(mask=nnunet_mask, dir_deriv_ct=dir_deriv_ct, spacing=spacing, center=center)

        print(f"\tPredicted: {fm:.4f}")
        print(f"\tGT: {doc:.4f}")
        print(f"\tROI: {nnunet:.4f}")
        print(f"\tDelta (predicted - GT): {(fm - doc):.4f} | Better: {(fm - doc < 0)}")

    
if __name__ == "__main__":
    main()