import os
import sys
import json
import math
import yaml
from copy import deepcopy

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from main import _process_one_patient, _save_3d_mask
import utils

output_dir_names = {
    0: "center_output",
    1: "pos_delta_output",
    2: "neg_delta_output"
}

def _get_final_threshold(config, thresholds, scores):
    method = config["multislice"]["threshold_aggregation_method"]

    weights = 1.0 / scores

    if method == "weighted_mean":
        return np.sum(weights * thresholds) / np.sum(weights)
    elif method == "weighted_median":
        return utils.get_weighted_median(
            weights=weights,
            values=thresholds
        )
    else:
        print(f"Unknown threshold aggregation method. Returning first threshold of the list.")
        return thresholds[0]

def _save_multislice_3d_mask(config, threshold, patient_id, patient_data):
    ct_path = patient_data["ct_path"]
    spacing, ct = utils.scan_to_np_array(ct_path, return_spacing=True)

    try:
        doc_mask_path = patient_data["doc_mask_path"]
        doc_mask = utils.scan_to_np_array(doc_mask_path)
        if patient_id == "patient_0001":
            doc_mask = doc_mask[:, :, :, 3]
        elif patient_id == "patient_0010":
            doc_mask = doc_mask[:, :, :, 1]
            doc_mask = (doc_mask == 1).astype(doc_mask.dtype)
        elif patient_id == "patient_0053":
            doc_mask = doc_mask[:, :, :, 2]
            doc_mask = (doc_mask == 1).astype(doc_mask.dtype)
        else:
            doc_mask = doc_mask[:, :, :, 1]
            doc_mask = (doc_mask == 2).astype(doc_mask.dtype)
    except Exception as e:
        print(f"Exception occured during doc mask loading. Setting mask to None.")
        print(f"Exception: {e}")
        return

    nnunet_mask_path = patient_data["nnunet_mask_path"]
    nnunet_mask_sitk, nnunet_mask = utils.scan_to_np_array(nnunet_mask_path, return_sitk=True)
    ventricle = (nnunet_mask == 3).astype(nnunet_mask.dtype)

    return _save_3d_mask(
        config=config,
        ct=ct,
        spacing=spacing,
        ventricle=ventricle,
        nnunet_mask_sitk=nnunet_mask_sitk,
        threshold=threshold,
        doc_mask=doc_mask
    )

def _process_one_patient_multislice(patient_id: str, patient_data: dict, config: dict):
    thresholds = []
    scores = []
    z_slices = {}
    patient_root = os.path.join(ROOT_DIR, "threshold_experiment_output", patient_id)
    
    for i in range(3):
        config["output_dir_name"] = os.path.join(patient_root, output_dir_names[i])

        center_output_dir = os.path.join(ROOT_DIR, "threshold_experiment_output", patient_id, "center_output")
        if os.path.exists(center_output_dir) and i != 0:
            center_results_path = os.path.join(center_output_dir, "results.json")
            with open(center_results_path, 'r') as f:
                center_results = json.load(f)
            
            z_data = center_results["preprocessing"]["z_data"]
            z_start = z_data["start"]
            z_middle = z_data["middle"]
            z_end = z_data["end"]

            multiplier = 1 if i % 2 == 1 else -1
            delta = max(1, math.ceil((z_end - z_start) / 10))
            new_z_middle = z_middle + multiplier * delta
            config["preprocessing"]["z_middle"] = new_z_middle

            z_slices["center"] = z_middle
            if multiplier == 1:
                z_slices["positive_delta"] = new_z_middle
            else:
                z_slices["negative_delta"] = new_z_middle

        _process_one_patient(
            patient_id=patient_id,
            patient_data=patient_data,
            config=config,
            make_output_dir=False
        )

        results_path = os.path.join(config["output_dir_name"], "results.json")
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        threshold = results["output_data"]["best_threshold"]
        score = results["postprocessing"][str(threshold)]["total_score"]

        thresholds.append(threshold)
        scores.append(score)

    thresholds = np.array(thresholds)
    scores = np.array(scores)

    final_threshold = _get_final_threshold(
        config=config,
        thresholds=thresholds,
        scores=scores
    )
    print(f"\tFinal threshold: {final_threshold}")
    
    mask_metrics = {}
    if config["multislice"]["save_3d"]:
        print(f"\tSaving 3D mask of final threshold...")
        
        config["output_dir_name"] = patient_root
        mask_metrics = _save_multislice_3d_mask(
            config=config,
            threshold=final_threshold,
            patient_id=patient_id,
            patient_data=patient_data
        )

    results = {}
    results["final_threshold"] = final_threshold
    results["slices"] = z_slices
    
    results["mask_metrics"] = mask_metrics

    with open(os.path.join(patient_root, "results.json"), 'w') as f:
        json.dump(results, f, indent=3)

def _process_multiple_patient_multislice(patients_to_process, patients_data, config):
    for patient_id in patients_to_process:
        print(f"Processing {patient_id}")
        if patient_id == "patient_0025":
            continue

        patient_config = deepcopy(config)
        patient_data = patients_data[patient_id]
        _process_one_patient_multislice(
            patient_id=patient_id,
            patient_data=patient_data,
            config=patient_config
        )

def main():
    with open(os.path.join(ROOT_DIR, "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)

    with open(os.path.join(ROOT_DIR, "patients_data.json"), 'r') as f:
        patients_data = yaml.safe_load(f)

    patient_id = "patient_0001"
    patient_data = patients_data[patient_id]
    _process_one_patient_multislice(
        patient_id=patient_id,
        patient_data=patient_data,
        config=config
    )

    # patients_to_process = []
    # with open(os.path.join(ROOT_DIR, "patients_to_process.txt"), 'r') as file:
    #     for line in file:
    #         line = line.strip()
    #         patients_to_process.append(line)
    # _process_multiple_patient_multislice(
    #     patients_to_process=patients_to_process,
    #     patients_data=patients_data,
    #     config=config
    # )

if __name__ == "__main__":
    main()