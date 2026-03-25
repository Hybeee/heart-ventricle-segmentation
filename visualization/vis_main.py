import numpy as np

import yaml
import os

from main import _process_one_patient
from visualization.plots import ViewData, VentricleData, ThresholdsData

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_thresholds_data(thresholds_dir: str, alg_results: dict) -> ThresholdsData:
    thresholds = []
    masks = []
    boundaries = []
    polar_masks = []
    polar_boundaries = []

    for threshold in alg_results["postprocessing"].keys():
        thresholds.append(threshold)

        mask = np.load(os.path.join(thresholds_dir, threshold, "mask.npy"))
        boundary = np.load(os.path.join(thresholds_dir, threshold, "mask_boundary.npy"))
        polar_mask = np.load(os.path.join(thresholds_dir, threshold, "polar_mask.npy"))
        polar_boundary = np.load(os.path.join(thresholds_dir, threshold, "polar_mask_boundary.npy"))

        masks.append(mask)
        boundaries.append(boundary)
        polar_masks.append(polar_mask)
        polar_boundaries.append(polar_boundary)
    
    thresholds_data = ThresholdsData(
        thresholds=thresholds,
        masks=masks,
        boundaries=boundaries,
        polar_masks=polar_masks,
        polar_boundaries=polar_boundaries
    )

    return thresholds_data

def _get_nms_results(results: dict):
    delta = 20
    top_n = 5

    anchor_points = []
    
    ret_results = {}
    for k in results.keys():
        if k == "postprocessing":
            continue
        ret_results[k] = results[k]
    
    ret_results["postprocessing"] = {}
    postprocessing_data = results["postprocessing"]

    for threshold in postprocessing_data.keys():
        if len(anchor_points) == top_n:
            break

        if len(anchor_points) == 0:
            anchor_points.append(threshold)
            ret_results["postprocessing"][threshold] = postprocessing_data[threshold]
            continue

        overlapping = False
        for anchor_point in anchor_points:
            if abs(float(threshold) - float(anchor_point)) <= delta:
                overlapping = True
                break
        
        if not overlapping:
            anchor_points.append(threshold)
            ret_results["postprocessing"][threshold] = postprocessing_data[threshold]

    return ret_results

def _load_alg_results(input_dir: str) -> dict:
    with open(os.path.join(input_dir, "results.json")) as f:
        results = yaml.safe_load(f)
    
    use_nms = True

    if use_nms:
        return _get_nms_results(results)
    else:
        return results

def _load_data(input_dir: str) -> ViewData:
    preproc_dir = os.path.join(input_dir, "preprocessing", "np")
    thresholds_dir = os.path.join(input_dir, "thresholds", "np")

    # np.load(os.path.join())

    ct = np.load(os.path.join(preproc_dir, "ct.npy"))
    polar_grad = np.load(os.path.join(preproc_dir, "polar_dir_grad.npy"))

    doc_mask = np.load(os.path.join(preproc_dir, "doc_mask.npy"))
    doc_mask_boundary = np.load(os.path.join(preproc_dir, "doc_mask_boundary.npy"))
    polar_doc_mask = np.load(os.path.join(preproc_dir, "polar_doc_mask.npy"))
    polar_doc_mask_boundary = np.load(os.path.join(preproc_dir, "polar_doc_mask_boundary.npy"))
    gt_data = VentricleData(
        mask=doc_mask,
        boundary=doc_mask_boundary,
        polar_mask=polar_doc_mask,
        polar_boundary=polar_doc_mask_boundary
    )

    ventricle = np.load(os.path.join(preproc_dir, "ventricle.npy"))
    ventricle_boundary = np.load(os.path.join(preproc_dir, "ventricle_boundary.npy"))
    polar_ventricle = np.load(os.path.join(preproc_dir, "polar_ventricle.npy"))
    polar_ventricle_boundary = np.load(os.path.join(preproc_dir, "polar_ventricle_boundary.npy"))
    nnunet_data = VentricleData(
        mask=ventricle,
        boundary=ventricle_boundary,
        polar_mask=polar_ventricle,
        polar_boundary=polar_ventricle_boundary
    )

    alg_results = _load_alg_results(input_dir)

    thresholds_data = _load_thresholds_data(
        thresholds_dir=thresholds_dir,
        alg_results=alg_results
    )

    view_data = ViewData(
        ct=ct, polar_grad=polar_grad,
        gt_data=gt_data, nnunet_data=nnunet_data,
        thresholds_data=thresholds_data,
        alg_results=alg_results
    )

    return view_data

def main():
    patient_id = "patient_0001"
    input_dir_name = "vis_output"
    input_dir = os.path.join(ROOT_DIR, input_dir_name, patient_id)

    if (not os.path.exists(input_dir)) or (not os.path.exists(os.path.join(input_dir, "preprocessing"))):
        with open(os.path.join(ROOT_DIR, "config.yaml"), 'r') as f:
            config = yaml.safe_load(f)

        config["cleanup"] = False
        config["output_dir_name"] = input_dir_name
        config["postprocessing"]["save_3d_mask"] = False
        config["postprocessing"]["include_detailed_valley_data"] = False

        with open(os.path.join(ROOT_DIR, "patients_data.json"), 'r') as f:
            patients_data = yaml.safe_load(f)
        
        patient_data = patients_data[patient_id]
        _process_one_patient(
            patient_id=patient_id,
            patient_data=patient_data,
            config=config
        )
    
    view_data = _load_data(input_dir=input_dir)
    

if __name__ == "__main__":
    main()