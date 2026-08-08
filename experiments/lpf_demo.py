import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

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

def _get_bbox_middle_slice_z(mask):
    z_dim = mask.shape[1]

    for z in range(z_dim):
        if np.any(mask[:, z, :] == 1):
            z_start = z
            break

    for z in reversed(range(z_dim)):
        if np.any(mask[:, z, :] == 1):
            z_end = z
            break

    z_middle = (z_start + z_end) // 2

    return (z_start, z_middle, z_end)

def smooth_dip(pb, start=150, end=200, depth = 24):
    r = pb[:, 0].astype(float).copy()
    t = np.linspace(0, np.pi, end - start)
    r[start:end] -= depth * np.sin(t)
    noisy_pb = np.column_stack((r, pb[:, 1].copy()))

    return noisy_pb

def ripple(pb, start=150, end=200, depth = 35):
    r = pb[:, 0].astype(float).copy()
    t = np.linspace(0, np.pi, end - start)
    window = np.sin(t)

    r[start:end] -= depth * window
    r[start:end] -= 3 * np.sin(8 * t) * window

    noisy_pb = np.column_stack((r, pb[:, 1].copy()))

    return noisy_pb

def lfp(pb, noisy_pb):
    r = pb[:, 0].astype(float).copy()
    noisy_r = noisy_pb[:, 0].astype(float).copy()

    R = np.fft.fft(r)
    noisy_R = np.fft.fft(noisy_r)

    plt.semilogy(np.abs(R), c='r')
    plt.semilogy(np.abs(noisy_R), c='b', alpha=0.5)
    plt.show()

    k = 7
    Rf = np.zeros_like(noisy_R)
    Rf[:k] = noisy_R[:k]
    Rf[-k:] = noisy_R[-k:]

    r_lfp = np.column_stack((np.real(np.fft.ifft(Rf)), pb[:, 1]))

    return r_lfp

def _get_cart_bp(bp, center):
    rs = bp[:, 0].astype(float).copy()
    thetas = bp[:, 1].astype(float).copy()
    thetas = np.deg2rad(thetas)

    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)

    cart_bp = center + np.stack((ys, xs), axis=-1)

    return cart_bp

def main():
    patient_id = "patient_0008"
    ct = utils.scan_to_np_array(
        f"C:\\BME\\mester\\2_felev\\onlab_2\\code\\nhakni\\solution\\postproc_alg_vars_output\\{patient_id}\\ct.nii.gz"
    )
    mask = utils.scan_to_np_array(
        f"C:\\BME\\mester\\2_felev\\onlab_2\\code\\nhakni\\solution\\postproc_alg_vars_output\\{patient_id}\\final_mask_nip.seg.nrrd"
    )

    z_start, z_middle, z_end = _get_bbox_middle_slice_z(mask=mask)
    z_middle = 229

    ct = np.flipud(ct[:, z_middle, :])
    mask = np.flipud(mask[:, z_middle, :])
    center = utils.calculate_slice_center(slice=mask)
    polar_converter = _create_polar_converter(
        center=center,
        ct=ct
    )

    polar_mask = polar_converter.cv2WarpPolar(image=mask.astype(np.int32))
    polar_mask = (polar_mask != 0).astype(polar_mask.dtype)

    pb = utils.get_polar_boundary_points(polar_mask=polar_mask, theta_step_size=1)

    noisy_pb = pb
    # noisy_pb = ripple(pb=noisy_pb, start=30, end=31)

    lfp_pb = lfp(pb=pb, noisy_pb=noisy_pb)

    h, w = polar_mask.shape
    plt.scatter(
        pb[:, 1],
        pb[:, 0],
        s=5,
        marker='o',
        c='blue',
        alpha=0.3
    )
    plt.scatter(
        noisy_pb[:, 1],
        noisy_pb[:, 0],
        s=5,
        marker='o',
        c='red',
        alpha=0.3
    )
    plt.scatter(
        lfp_pb[:, 1],
        lfp_pb[:, 0],
        s=5,
        marker='o',
        c='green',
        alpha=0.3
    )

    plt.xlim(0, w)
    plt.ylim(h, 0)
    plt.gca().set_aspect('equal')
    plt.show()

    cart_bp = _get_cart_bp(bp=lfp_pb, center=center)

    plt.imshow(ct, cmap='gray')
    plt.imshow(mask, cmap='Blues', alpha=0.2)
    # plt.scatter(
    #     cart_bp[:, 1],
    #     cart_bp[:, 0],
    #     s=1,
    #     c='green',
    #     alpha=0.5
    # )
    plt.scatter(
        center[1],
        center[0],
        s=3,
        c='red',
        marker='x',
        alpha=0.5
    )
    plt.show()

if __name__ == "__main__":
    main()