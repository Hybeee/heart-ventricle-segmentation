import SimpleITK as sitk
import numpy as np

import os
import sys
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils


def geometries_match(ct, mask, tol=1e-3):
    return (
        ct.GetSize() == mask.GetSize()
        and all(abs(a - b) < tol for a, b in zip(ct.GetSpacing(), mask.GetSpacing()))
        and all(abs(a - b) < tol for a, b in zip(ct.GetOrigin(), mask.GetOrigin()))
        and ct.GetDirection() == mask.GetDirection()
    )

def get_roi(mask_sitk):
    mask = sitk.GetArrayFromImage(mask_sitk)
    coords = np.argwhere(mask)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    # option A: padding LV bbox with bbox ratio
    # ratio = 0.1
    # diffs = maxs - mins
    # pad = diffs * ratio

    # option B: padding LV bbox with fix pad in mm
    padding_mm = 5.0
    spacing_xyz = mask_sitk.GetSpacing()
    pad = [ # xyz -> zyx
        int(np.ceil(padding_mm / spacing_xyz[2])),
        int(np.ceil(padding_mm / spacing_xyz[1])),
        int(np.ceil(padding_mm / spacing_xyz[0]))
    ]

    shape = np.array(mask.shape)
    mins = np.clip(mins - pad, 0, shape - 1)
    maxs = np.clip(maxs + pad, 0, shape - 1)

    z_min, y_min, x_min = mins
    z_max, y_max, x_max = maxs

    roi_index = [int(x_min), int(y_min), int(z_min)]
    roi_size = [int(x_max - x_min + 1), int(y_max - y_min + 1), int(z_max - z_min + 1)]

    return roi_index, roi_size

def main():
    data_dir = os.path.join(ROOT_DIR, "pipeline_output")
    label_dir = os.path.join(ROOT_DIR, "heart_muscle_segmentation_output")

    data_patient_ids = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    label_patient_ids = sorted([d for d in os.listdir(label_dir) if os.path.isdir(os.path.join(label_dir, d))])

    if data_patient_ids != label_patient_ids:
        missing_in_labels = set(data_patient_ids) - set(label_patient_ids)
        missing_in_data = set(label_patient_ids) - set(data_patient_ids)
        raise RuntimeError(
            f"Data/label mismatch. Missing in labels: {missing_in_labels}, "
            f"missing in data: {missing_in_data}"
        )

    patient_ids = data_patient_ids

    dest_dir = os.path.join(ROOT_DIR, "nnunet_finetune", "nnUNet_raw")
    os.makedirs(dest_dir, exist_ok=True)

    dataset_name = "Dataset001_Proto"
    dataset_dir = os.path.join(dest_dir, dataset_name)
    if os.path.exists(dataset_dir):
        raise RuntimeError(f"Dataset `{dataset_name}` already exists.")

    folders = ["imagesTr", "labelsTr"]
    for folder in folders:
        os.makedirs(os.path.join(dataset_dir, folder), exist_ok=True)

    for patient_id in patient_ids:
        print(f"Copying {patient_id}...")

        ct_src = os.path.join(data_dir, patient_id, "ct.nii.gz")
        ct_sitk = sitk.ReadImage(ct_src)
        label_src = os.path.join(label_dir, patient_id, "mask_hull.seg.nrrd")
        label_seg_nrrd = sitk.ReadImage(label_src)

        if not geometries_match(ct_sitk, label_seg_nrrd):
            print(f"Geometries didn't match for {patient_id}. Resampling label onto CT grid...")
            label_seg_nrrd = sitk.Resample(
                label_seg_nrrd, ct_sitk, sitk.Transform(),
                sitk.sitkNearestNeighbor, 0, label_seg_nrrd.GetPixelID()
            )

        ct_dst = os.path.join(dataset_dir, "imagesTr", f"{patient_id}_0000.nii.gz")
        shutil.copy(ct_src, ct_dst)
        sitk.WriteImage(label_seg_nrrd, os.path.join(dataset_dir, "labelsTr", f"{patient_id}.nii.gz"))

if __name__ == "__main__":
    main()