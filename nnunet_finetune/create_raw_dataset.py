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