import numpy as np
import SimpleITK as sitk

import os
import sys
import shutil
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

SEED = 42
import numpy as np
rng = np.random.default_rng(seed=SEED)

def get_train_test_split(patients_all):
    n_tests = 5
    patients_ts = sorted(rng.choice(patients_all, n_tests, replace=False).tolist())
    patients_tr = [pid for pid in patients_all if pid not in patients_ts]

    return patients_tr, patients_ts

def create_splits_final(new_dataset_name, nnunet_dir, orig_dataset_name):
    preprocessed_dir = os.path.join(nnunet_dir, "nnUNet_preprocessed")
    os.makedirs(os.path.join(preprocessed_dir, new_dataset_name), exist_ok=True)

    with open(os.path.join(preprocessed_dir, orig_dataset_name, "splits_final.json")) as f:
        splits_data = json.load(f)

    new_splits_data = []
    test_data = []

    n_tests = 5
    used = set()
    for fold in splits_data:
        tr = fold["train"]
        val = fold["val"]
        candidates = [p for p in tr if p not in used]
        ts = sorted(rng.choice(candidates, n_tests, replace=False).tolist())
        used.update(ts)
        tr = [p for p in tr if p not in ts]
        new_splits_data.append({
            "train": tr,
            "val": val
        })
        test_data.append(ts)

    with open(os.path.join(preprocessed_dir, new_dataset_name, "splits_final.json"), 'w') as f:
        json.dump(new_splits_data, f, indent=2)

    with open(os.path.join(preprocessed_dir, new_dataset_name, "per_fold_test_cases.json"), 'w') as f:
        json.dump(test_data, f, indent=2)

def _get_mask_image(patient_id, patient_data):
    doc_mask_path = patient_data["doc_mask_path"]
    doc_mask_np = utils.scan_to_np_array(doc_mask_path)

    if patient_id == "patient_0005":
        structures = ["leftccavity", "papillaryPM", "papillaryAL"]
    else:
        structures = ["leftcavity", "papillaryPM", "papillaryAL"]
    
    data = {}

    orig_img = sitk.ReadImage(doc_mask_path)
    metadata = orig_img.GetMetaDataKeys()

    for i, structure in enumerate(structures):
        for key in metadata:
            if key.endswith("_Name") and orig_img.GetMetaData(key) == structure:
                segment_num = key.split("_")[0].replace("Segment", "")

                layer = int(orig_img.GetMetaData(f"Segment{segment_num}_Layer"))
                label = int(orig_img.GetMetaData(f"Segment{segment_num}_LabelValue"))

                data[structure] = {
                    "index": i+1,
                    "layer": layer,
                    "label": label
                }

    mask = np.zeros(doc_mask_np.shape[:3], dtype=np.uint8)
    for structure in structures:
        index = data[structure]["index"]
        layer = data[structure]["layer"]
        label = data[structure]["label"]
        mask[doc_mask_np[..., layer] == label] = index

    mask_image = sitk.GetImageFromArray(mask)
    mask_image.CopyInformation(orig_img)

    return mask_image

def geometries_match(ct, mask, tol=1e-3):
    return (
        ct.GetSize() == mask.GetSize()
        and all(abs(a - b) < tol for a, b in zip(ct.GetSpacing(), mask.GetSpacing()))
        and all(abs(a - b) < tol for a, b in zip(ct.GetOrigin(), mask.GetOrigin()))
        and ct.GetDirection() == mask.GetDirection()
        )

def copy_to_raw_dataset(patient_ids, patients_data, subset, cts_dir, out_dir):
    for patient_id in patient_ids:
        print(f"({subset}) - Processing {patient_id}...")
        
        ct_path = os.path.join(cts_dir, patient_id, "ct.nii.gz")
        ct_image = sitk.ReadImage(ct_path)

        mask_image = _get_mask_image(
            patient_id=patient_id, 
            patient_data=patients_data[patient_id]
            )
        before = np.bincount(sitk.GetArrayFromImage(mask_image).ravel(), minlength=4)

        if not geometries_match(ct_image, mask_image):
            mask_image = sitk.Resample(
                mask_image, ct_image, sitk.Transform(),
                sitk.sitkNearestNeighbor, 0, mask_image.GetPixelID()
            )

        after = np.bincount(sitk.GetArrayFromImage(mask_image).ravel(), minlength=4)
        assert np.array_equal(before, after), f"{patient_id}: label counts changed {before} -> {after}"

        shutil.copy(
            src=ct_path,
            dst=os.path.join(out_dir, f"images{subset}", f"{patient_id}_0000.nii.gz")
        )
        sitk.WriteImage(
            image=mask_image,
            fileName=os.path.join(out_dir, f"labels{subset}", f"{patient_id}.nii.gz")
        )

def main():
    with open(os.path.join(ROOT_DIR, "patients_data.json")) as f:
        patients_data = json.load(f)

    data_dir = os.path.join(ROOT_DIR, "postproc_alg_vars_output")
    nnunet_dir = os.path.join(ROOT_DIR, "nnunet_finetune")

    patients_to_skip = [
        'patient_0013',
        'patient_0025',
        'patient_0028',
        'patient_0032',
        'patient_0034',
        'patient_0043',
        'patient_0046',
        'patient_0053',
        'patient_0054'
    ]

    dataset_name = "Dataset005_LV_HM_Doc_Ts_Per_Fold"
    raw_dir = os.path.join(nnunet_dir, "nnUNet_raw", dataset_name)
    os.makedirs(raw_dir)

    patients_all = [pid for pid in patients_data.keys() if pid not in patients_to_skip]

    patients_ts = None
    # NO TEST SET
    # patients_tr = patients_all

    # GLOBAL TEST SET
    # patients_tr, patients_ts = get_train_test_split(patients_all)

    # PER-FOLD TEST SET
    patients_tr = patients_all
    orig_dataset_name = "Dataset003_LV_HM_Doc"
    create_splits_final(
        new_dataset_name=dataset_name,
        nnunet_dir=nnunet_dir,
        orig_dataset_name=orig_dataset_name,
    )

    sub_folders = ["imagesTr", "labelsTr"]
    if patients_ts is not None:
        sub_folders.extend(["imagesTs", "labelsTs"])
    for sub_folder in sub_folders:
        os.makedirs(os.path.join(raw_dir, sub_folder), exist_ok=True)

    copy_to_raw_dataset(
        patient_ids=patients_tr,
        patients_data=patients_data,
        subset="Tr",
        cts_dir=data_dir,
        out_dir=raw_dir
    )

    if patients_ts is not None:
        copy_to_raw_dataset(
            patient_ids=patients_ts,
            patients_data=patients_data,
            subset="Ts",
            cts_dir=data_dir,
            out_dir=raw_dir
        )

    dataset_info = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "LV": 1,
            "papillaryPM": 2,
            "papillaryAL": 3
        },
        "numTraining": len(patients_tr),
        "file_ending": ".nii.gz"
    }

    with open(os.path.join(raw_dir, "dataset.json"), 'w') as f:
        json.dump(dataset_info, f, indent=2)

if __name__ == "__main__":
    main()