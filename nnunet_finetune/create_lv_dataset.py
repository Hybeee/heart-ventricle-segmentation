"""
Uses LV segmentation given by heart_dataset
"""

import SimpleITK as sitk
import numpy as np

import os
import json
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

def _get_fixed_gt(ct_sitk, nnunet_ventricle, label):
    nnunet_ventricle = nnunet_ventricle.astype(bool)
    label = label.astype(bool)

    label_fixed = np.array(label & nnunet_ventricle, dtype=np.uint8)
    label = sitk.GetImageFromArray(label_fixed)
    label.CopyInformation(ct_sitk)

    return label

def main():
    with open(os.path.join(ROOT_DIR, "patients_data.json")) as f:
        patients_data = json.load(f)

    data_dir = os.path.join(ROOT_DIR, "postproc_alg_vars_output")

    patient_ids = os.listdir(data_dir)
    patients_to_skip = [
        'patient_0013',
        'patient_0028',
        'patient_0032',
        'patient_0034',
        'patient_0043',
        'patient_0046',
        'patient_0053',
    ]

    dest_dir = os.path.join(ROOT_DIR, "nnunet_finetune", "nnUNet_raw")
    os.makedirs(dest_dir, exist_ok=True)

    dataset_name = "Dataset002_LV_Doc"
    dataset_dir = os.path.join(dest_dir, dataset_name)

    folders = ["imagesTr", "labelsTr"]
    for folder in folders:
        os.makedirs(os.path.join(dataset_dir, folder), exist_ok=True)

    for patient_id in patient_ids:
        patient_data = patients_data[patient_id]
        if patient_id in patients_to_skip:
            continue

        print(f"Copying {patient_id}...")

        ct_src = os.path.join(data_dir, patient_id, "ct.nii.gz")
        ct_sitk = sitk.ReadImage(ct_src)
        label_src = os.path.join(data_dir, patient_id, "doc_mask.seg.nrrd")
        label_seg_nrrd = sitk.ReadImage(label_src)
        nnunet_mask_path = patient_data["nnunet_mask_path"]
        nnunet_mask = utils.scan_to_np_array(nnunet_mask_path)
        nnunet_ventricle = (nnunet_mask == 3).astype(nnunet_mask.dtype)

        
        label_seg_np = sitk.GetArrayFromImage(label_seg_nrrd)
        label_seg_nrrd = _get_fixed_gt(ct_sitk, nnunet_ventricle, label_seg_np)

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