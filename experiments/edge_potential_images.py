import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import time
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

SIGMAS = [0.5, 1.0, 2.0, 5.0, 10.0]

def laplacian(signed_distance_np, sigma):
    return 1.0 - np.exp(-np.abs(signed_distance_np) / sigma)

def gaussian(signed_distance_np, sigma):
    return 1.0 - np.exp(-(signed_distance_np ** 2) / (2.0 * sigma ** 2))

def _get_edge_potential(mask_sitk, func=laplacian, sigma=2.0):
    signed_distance = sitk.SignedMaurerDistanceMap(
        mask_sitk,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True
    )

    signed_distance_np = sitk.GetArrayFromImage(signed_distance)
    edge_potential_np = func(signed_distance_np, sigma)

    return edge_potential_np

def _process_patient(data_dir, patient_id, output_dir):
    patient_dir = os.path.join(data_dir, patient_id)
    patient_output_dir = os.path.join(output_dir, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)

    files_to_copy = ["ct.nii.gz", "final_mask_nip.seg.nrrd"]

    for file in files_to_copy:
        shutil.copyfile(
            src=os.path.join(patient_dir, file),
            dst=os.path.join(patient_output_dir, file)
        )

    mask_sitk, mask = utils.scan_to_np_array(scan_path=os.path.join(patient_dir, "final_mask_nip.seg.nrrd"), return_sitk=True)

    functions = {
        "laplacian": laplacian,
        "gaussian": gaussian
    }

    for function_name, function in functions.items():
        curr_output_dir = os.path.join(patient_output_dir, function_name)
        os.makedirs(curr_output_dir, exist_ok=True)
        
        for sigma in SIGMAS:
            edge_potential_np = _get_edge_potential(
                mask_sitk=mask_sitk,
                func=function,
                sigma=sigma
            )

            name = f"ep_{sigma}"
            utils.save_data(
                data=edge_potential_np,
                ref_sitk=mask_sitk,
                output_dir=curr_output_dir,
                name=name,
                is_mask=False
            )

def plot(curr_l_mask, curr_g_mask, slice_index, direction, dir_index):
    print(f"Mask shape: {curr_l_mask.shape}")

    try:
        l_coronal = np.flipud(curr_l_mask[:, slice_index, :])
        g_coronal = np.flipud(curr_g_mask[:, slice_index, :])
    except Exception as e:
        print(f"Invalid slice_index: {slice_index}. Mask shape: {curr_l_mask.shape}")
        return

    if direction.lower() not in ["row", "col"]:
        print(f"Invalid direction: {direction}.")
        return

    if direction.lower() == "row":
        try:
            l_profile = l_coronal[dir_index, :]
            g_profile = g_coronal[dir_index, :]
        except:
            print(f"Invalid {direction.lower()} index. Mask shape: {curr_l_mask.shape}")
            return
    else:
        try:
            l_profile = l_coronal[:, dir_index]
            g_profile = g_coronal[:, dir_index]
        except:
            print(f"Invalid {direction.lower()} index. Mask shape: {curr_l_mask.shape}")
            return

    threshold = 1.0
    pad = 20
    epsilon = 1e-3
    below = np.where((l_profile < threshold - epsilon) | (g_profile < threshold - epsilon))[0]

    if len(below) == 0:
        start, end = 0, len(l_profile)
    else:
        first, last = below[0], below[-1]
        start = max(0, first - pad)
        end = min(len(l_profile), last + pad + 1)

    x_axis = np.arange(start, end)
    l_profile = l_profile[start:end]
    g_profile = g_profile[start:end]

    print(f"{start} - {end}")

    fig, (ax_img, ax_profile) = plt.subplots(1, 2, figsize=(12, 5))

    ax_img.imshow(l_coronal, cmap='gray')
    if direction.lower() == "row":
        ax_img.axhline(y=dir_index, color="yellow", linewidth=1)
    else:
        ax_img.axvline(x=dir_index, color="yellow", linewidth=1)
    ax_img.set_title(f"Coronal slice (slice_index={slice_index})")

    ax_profile.plot(x_axis, l_profile, marker='o', markersize=3, color='r', linewidth=1, label="Laplacian")
    ax_profile.plot(x_axis, g_profile, marker='x', markersize=3, color='g', linewidth=1, label="Gaussian")
    ax_profile.set_xlabel(f"{direction.lower()} index")
    ax_profile.set_ylabel("edge potential")
    ax_profile.grid(alpha=0.3)
    ax_profile.legend()

    plt.tight_layout()
    plt.show()


def view_profile(data_dir):
    curr_sigma = None
    curr_l_mask = None
    curr_g_mask = None

    while True:
        user_input = input("Enter sigma,slice_index,row/col,dir_index... ")

        if user_input.lower() == "exit":
            break

        sigma, slice_index, direction, dir_index = user_input.split(",")
        sigma = float(sigma)
        slice_index = int(slice_index)
        direction = direction.strip()
        dir_index = int(dir_index)

        if curr_sigma is None or curr_sigma != sigma:
            curr_sigma = sigma

            curr_l_mask = utils.scan_to_np_array(os.path.join(data_dir, "laplacian", f"ep_{sigma}.nii.gz"))
            curr_g_mask = utils.scan_to_np_array(os.path.join(data_dir, "gaussian", f"ep_{sigma}.nii.gz"))

        plot(
            curr_l_mask=curr_l_mask,
            curr_g_mask=curr_g_mask,
            slice_index=slice_index,
            direction=direction,
            dir_index=dir_index
        )

def main():
    data_dir = os.path.join(f"{ROOT_DIR}", "pipeline_output")
    patient_id = "patient_0001"

    output_dir = os.path.join(f"{ROOT_DIR}", "edge_potential_output")
    os.makedirs(output_dir, exist_ok=True)

    # _process_patient(
    #     data_dir=data_dir,
    #     patient_id=patient_id,
    #     output_dir=output_dir
    # )

    view_profile(os.path.join(output_dir, patient_id))

if __name__ == "__main__":
    main()