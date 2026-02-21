import numpy as np
import matplotlib.pyplot as plt
import cv2

import os
import json

import utils

def _get_middle_slice(nnunet_mask):
    z_dim = nnunet_mask.shape[0]

    for z in range(z_dim):
        if np.any(nnunet_mask[z, :, :] == 1):
            z_start = z
            break

    for z in reversed(range(z_dim)):
        if np.any(nnunet_mask[z, :, :] == 1):
            z_end = z
            break

    z_middle = (z_start + z_end) // 2

    return z_middle

def _create_polar_converter(center, ct):
    maxRadius = min(
        center[0],
        center[1],
        ct.shape[0] - center[0] - 50,
        ct.shape[1] - center[1] - 50
    )

    radius_samples = maxRadius
    angle_samples = 360
    polar_converter = utils.PolarConverter(
        dsize=(radius_samples, angle_samples),
        maxRadius=maxRadius,
        center=(center[1], center[0]),
        flags=cv2.WARP_POLAR_LINEAR
    )

    return polar_converter

def _preprocess_ct(config,
                   ct, ct_spacing, center,
                   polar_converter: utils.PolarConverter):
    sigma = utils.get_scaled_sigma(
        curr_spacing=ct_spacing,
        ref_pixel_sigma=config['ref_sigma'],
        ref_spacing=config['ref_spacing']
    )

    if sigma[0] == sigma[1]:
        sigma = np.round(sigma[0], 2)
    else:
        print("Different sigma. Skipping.")
        return None

    _, polar_dir_grad = utils.get_directional_grad(
        ct=ct,
        center=center,
        sigma=sigma,
        polarConverter=polar_converter
    )

    polar_dir_grad = utils.clip_polar_gradients(polar_dir_grad_ct=polar_dir_grad,
                                                grad_clip=config['grad_clip'])
    
    return polar_dir_grad, sigma

def _preprocess_mask(config,
                     mask, polar_converter: utils.PolarConverter,
                     return_boundary=True):
    polar_mask = polar_converter.cv2WarpPolar(image=mask.astype(np.int32))
    polar_mask = (polar_mask != 0).astype(polar_mask.dtype)

    if return_boundary:
        polar_mask_boundary = utils.get_polar_boundary_points(polar_mask=polar_mask, theta_step_size=config['theta_step_size'])
        return polar_mask, polar_mask_boundary

    return polar_mask

def _get_mask_indices(polar_mask):
    theta_indices = np.where(polar_mask.any(axis=0))[0]
    if theta_indices.size == 0:
        return np.array([[-1, -1]])

    splits = np.where(np.diff(theta_indices) > 1)[0] + 1
    groups = np.split(theta_indices, splits)
    intervals = [[group[0], group[-1]] for group in groups]

    return np.array(intervals)

def _save_np(output_dir, array, name):
    np.save(os.path.join(output_dir, f"{name}.npy"), array)

def preprocess(config: dict,
               ct, ct_spacing,
               atrium, ventricle, lung,
               output_dir,
               doc_mask=None):
    
    """
    NOTE: atrium es ventricle a bemenet -> legyen a doc_mask is - ha adott - mar filterezett - ergo csak a kamrat tartalmazza
    """
    output_dir = os.path.join(output_dir, "preprocessing")
    os.makedirs(output_dir, exist_ok=True)
    output_dir_plots = os.path.join(output_dir, "plots")
    os.makedirs(output_dir_plots, exist_ok=True)
    output_dir_np = os.path.join(output_dir, "np")
    os.makedirs(output_dir_np, exist_ok=True)

    z_middle = _get_middle_slice(nnunet_mask=ventricle)

    ct = ct[z_middle, :, :]
    if doc_mask is not None:
        doc_mask = doc_mask[z_middle, :, :]
        doc_mask_boundary = utils.get_mask_boundary(doc_mask)
    atrium = atrium[z_middle, :, :]
    lung = lung[z_middle, :, :]
    ventricle = ventricle[z_middle, :, :]
    ventricle_boundary = utils.get_mask_boundary(ventricle)

    center = utils.calculate_slice_center(ventricle)

    if config['save_plots']:
        plt.imshow(ct, cmap='gray')
        plt.scatter(center[1], center[0], s=1, c='green')
        plt.scatter(
            ventricle_boundary[:, 1],
            ventricle_boundary[:, 0],
            s=0.1,
            c='b'
        )
        if doc_mask is not None:
            plt.scatter(
                doc_mask_boundary[:, 1],
                doc_mask_boundary[:, 0],
                s=0.1,
                c='r'
            )
        plt.axis('off')
        plt.savefig(os.path.join(output_dir_plots, "ct_with_masks.png"))
        plt.close()

    polar_converter = _create_polar_converter(center=center, ct=ct)

    polar_dir_grad, sigma = _preprocess_ct(
        config=config,
        ct=ct,
        ct_spacing=ct_spacing,
        center=center,
        polar_converter=polar_converter
    )

    if polar_dir_grad is None:
        return False
    
    polar_atrium = _preprocess_mask(
        config=config,
        mask=atrium,
        polar_converter=polar_converter,
        return_boundary=False
    )
    atrium_intervals = _get_mask_indices(polar_mask=polar_atrium)
    atrium_intervals = atrium_intervals[0]
    
    polar_lung = _preprocess_mask(
        config=config,
        mask=lung,
        polar_converter=polar_converter,
        return_boundary=False
    )
    lung_intervals =  _get_mask_indices(polar_mask=polar_lung)

    polar_ventricle, polar_ventricle_boundary = _preprocess_mask(
        config=config,
        mask=ventricle,
        polar_converter=polar_converter
    )

    if doc_mask is not None:
        polar_doc_mask, polar_doc_mask_boundary = _preprocess_mask(
            config=config,
            mask=doc_mask,
            polar_converter=polar_converter
        )
    
    if config['save_plots']:
        plt.imshow(polar_dir_grad, cmap='jet')
        plt.scatter(
            polar_ventricle_boundary[:, 1],
            polar_ventricle_boundary[:, 0],
            s=5,
            marker='o',
            c='blue',
            alpha=0.3,
        )
        if doc_mask is not None:
            plt.scatter(
                polar_doc_mask_boundary[:, 1],
                polar_doc_mask_boundary[:, 0],
                s=5,
                marker='o',
                c='red',
                alpha=0.3,
            )

        plt.savefig(os.path.join(output_dir_plots, "polar_plot.png"))
        plt.close()

    _save_np(output_dir_np, ct, "ct")
    _save_np(output_dir_np, ventricle, "ventricle")
    _save_np(output_dir_np, polar_dir_grad, "polar_dir_grad")
    _save_np(output_dir_np, polar_ventricle, "polar_ventricle")
    _save_np(output_dir_np, polar_ventricle_boundary, "polar_ventricle_boundary")
    if doc_mask is not None:
        _save_np(output_dir_np, doc_mask, "doc_mask")
        _save_np(output_dir_np, doc_mask_boundary, "doc_mask_boundary")
        _save_np(output_dir_np, polar_doc_mask, "polar_doc_mask")
        _save_np(output_dir_np, polar_doc_mask_boundary, "polar_doc_mask_boundary")

    data = {
        "middle_slice_index": z_middle,
        "center": center,
        "atrium_data": {
            "start": int(atrium_intervals[0]),
            "end": int(atrium_intervals[1])
        },
        "lung_data": {
            "starts": [int(x) for x in lung_intervals[:, 0]],
            "ends": [int(x) for x in lung_intervals[:, 1]]
        },
        "sigma": sigma
    }

    with open(os.path.join(output_dir, "data.json"), 'w') as f:
        json.dump(data, f, indent=2)

    return polar_converter