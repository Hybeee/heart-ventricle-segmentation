import SimpleITK as sitk
import numpy as np

import os
import yaml
import json

import utils

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def _get_masks(ct, nnunet_mask, threshold1, threshold2):
    threshold1 = float(threshold1)
    threshold2 = float(threshold2)

    mask1, mask2 = np.zeros_like(ct), np.zeros_like(ct)
    
    mask1[ct > threshold1] = 1
    mask1[nnunet_mask == 0] = 0

    mask2 = [ct > threshold2] = 1
    mask2[nnunet_mask == 0] = 0

    return mask1, mask2

def _get_mask_similarity(mask1, mask2):
    numerator = np.abs(mask1 - mask2).sum()
    denominator = (mask1 + mask2).sum() / 2

    if denominator == 0:
        print("Both masks were empty")
        return 0.0
    
    return numerator / denominator

def _get_patient_info(patient_data, patient_id, dir1, dir2):
    ct = utils.scan_to_np_array(patient_data["ct_path"])
    nnunet_mask = utils.scan_to_np_array(patient_data["nnunet_mask_path"])

    with open(os.path.join(dir1, patient_id, "result.json")) as f:
        result1 = json.load(f)
    with open(os.path.join(dir2, patient_id, "result.json")) as f:
        result2 = json.load(f)

    first_data = {
        "threshold": result1["best_threshold"],
        "middle_slice": result1["preprocessing"]["middle_slice_index"]
    }

    second_data = {
        "threshold": result2["best_threshold"],
        "middle_slice": result2["preprocessing"]["middle_slice_index"]
    }

    mask1, mask2 = _get_masks(ct, nnunet_mask, first_data["threshold"], second_data["threshold"])

    metric = _get_mask_similarity(mask1, mask2)

    threshold_data = {
        "m1": result1["best_threshold"],
        "m2": result2["best_thresholds"],
        "abs_diff": np.abs(result1["best_threshold"] - result2["best_threshold"])
    }

    middle_slice_data = {
        "m1": result1["preprocessing"]["middle_slice_index"],
        "m2": result2["preprocessing"]["middle_slice_index"],
        "abs_diff": np.abs(result1["preprocessing"]["middle_slice_index"] - result2["preprocessing"]["middle_slice_index"])
    }

    result = {
        "score": metric,
        "threshold_data": threshold_data,
        "middle_slice_data": middle_slice_data
    }

    return result

def main():
    output_dir = os.path.join(ROOT_DIR, "mask_similarity_analysis")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(ROOT_DIR, "patients_data.json")) as f:
        patients_data = yaml.safe_load(f)

    patients_to_process = []
    with open(os.path.join(ROOT_DIR, "patients_to_process.txt"), 'r') as f:
        for line in f:
            line = line.strip
            patients_to_process.append(line)

    name1= "bbox_middle"
    dir1 = os.path.join(ROOT_DIR, "other_test_output_bbox_middle_old_w0")

    name2 = "com"
    dir2 = os.path.join(ROOT_DIR, "other_test_output_com_old_w0")

    if not (os.path.exists(dir1) and os.path.exists(dir2)):
        print(f"One of the given directories does not exist!")
        return
    
    for patient_id in patients_to_process:
        data = _get_patient_info(patients_data[patient_id], patient_id, dir1, dir2)

        data = {
            "names": {
                "mask1": name1,
                "mask2": name2
            },
            **data
        }

        with open(os.path.join(os.path.join(output_dir, "data.json"))) as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()