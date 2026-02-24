import numpy as np
import matplotlib.pyplot as plt

import os
import json

def get_nms_radius(config, score_data):
    delta = config["nms_radius_delta"]

    anchor_points = []
    nms_result = {}

    for threshold in score_data:
        if len(anchor_points) == 0:
            anchor_points.append(score_data[threshold]["avg_r"])
            nms_result[threshold] = score_data[threshold]
            continue
            
        overlapping = False
        for anchor_point in anchor_points:
            curr_avg_r = score_data[threshold]["avg_r"]
            if abs(curr_avg_r - anchor_point) <= delta:
                overlapping = True
                break
        
        if not overlapping:
            anchor_points.append(score_data[threshold]["avg_r"])
            nms_result[threshold] = score_data[threshold]

    return nms_result

def get_nms_threshold(config, score_data):
    delta = config["nms_threshold_delta"]

    anchor_points = []
    nms_result = {}

    for threshold in score_data:
        if len(anchor_points) == 0:
            anchor_points.append(threshold)
            nms_result[threshold] = score_data[threshold]
            continue

        overlapping = False
        for anchor_point in anchor_points:
            if abs(float(threshold) - float(anchor_point)) <= delta:
                overlapping = True
                break

        if not overlapping:
            anchor_points.append(threshold)
            nms_result[threshold] = score_data[threshold]

    return nms_result

def get_nms_result(config, score_data, nms_type, output_dir):
    if nms_type == "radius":
        nms_result = get_nms_radius(config=config, score_data=score_data)
    elif nms_type == "threshold":
        nms_result = get_nms_threshold(config=config, score_data=score_data)
    else:
        nms_result = None

    if nms_result:
        with open(os.path.join(output_dir, "nms_result.json"), 'w') as f:
            json.dump(nms_result, f, indent=2)
    
    return nms_result

def _get_valley_indices(polar_grad: np.ndarray, boundary_points):
    fo_deriv = polar_grad
    so_deriv = np.gradient(fo_deriv, axis=0)

    zero_crossings = np.diff(np.signbit(so_deriv), axis=0)

    slope = so_deriv[1:, :] - so_deriv[:-1, :]
    valley_mask = zero_crossings & (slope > 0)
    valley_index_map = np.cumsum(valley_mask, axis=0)
    valley_index_map = np.pad(valley_index_map, ((1,0), (0,0)), mode='constant')

    r_idx, theta_idx = boundary_points[:, 0], boundary_points[:, 1]
    assigned_valleys = valley_index_map[r_idx, theta_idx]

    first_valley = np.argmax(valley_mask, axis=0)
    last_valley = valley_mask.shape[0] - 1 - np.argmax(valley_mask[::-1, :], axis=0)
    valid = (r_idx >= first_valley[theta_idx]) & (r_idx <= last_valley[theta_idx])

    assigned_valleys[~valid] = -1

    return assigned_valleys

def dummy_decider(nms_result: dict, data_dict: dict, thresholds_dir: str,
                  polar_grad: np.ndarray, polar_ventricle_b: np.ndarray):
    best_threshold = None
    best_score = -np.inf

    for threshold in nms_result.keys():
        threshold_dir = os.path.join(thresholds_dir, threshold)
        polar_threshold_b = np.load(os.path.join(threshold_dir, "polar_mask_boundary.npy"))
        r_indices = polar_threshold_b[:, 0].astype(int)
        theta_indices = polar_threshold_b[:, 1].astype(int)

        lung_data = data_dict["lung_data"]
        starts = lung_data["starts"]
        ends = lung_data["ends"]

        valid_mask = (
            ~((theta_indices >= ends[0]) & (theta_indices <= starts[1]))
            ) & (r_indices != -1)
        r_indices = r_indices[valid_mask]
        theta_indices = theta_indices[valid_mask]

        total = np.abs(polar_grad[r_indices, theta_indices]).sum()

        assigned_valleys_t = _get_valley_indices(
            polar_grad=polar_grad,
            boundary_points=polar_threshold_b)
        assigned_valleys_v = _get_valley_indices(
            polar_grad=polar_grad,
            boundary_points=polar_ventricle_b
        )

        if best_score is None:
            best_score = total
            best_threshold = threshold

        if np.any(assigned_valleys_t == assigned_valleys_v):
            print(f"Skipping {threshold}.")
            continue

        print(f"\t{threshold}: {total}")
        nms_result[threshold]["total_depth"] = float(total)

        if total > best_score:
            best_score = total
            best_threshold = threshold

    return best_threshold, nms_result


def _get_input(input_dir):
    with open(os.path.join(input_dir, "thresholds", "score_data.json"), 'r') as f:
        score_data = json.load(f)

    with open(os.path.join(input_dir, "preprocessing", "data.json"), 'r') as f:
        data_dict = json.load(f)

    thresholds_dir = os.path.join(input_dir, "thresholds", "np")

    polar_grad = np.load(os.path.join(input_dir, "preprocessing", "np", "polar_dir_grad.npy"))

    polar_ventricle_b = np.load(os.path.join(input_dir, "preprocessing", "np", "polar_ventricle_boundary.npy"))

    return score_data, data_dict, thresholds_dir, polar_grad, polar_ventricle_b

def postprocess(config, output_dir):

    score_data, data_dict, thresholds_dir, polar_grad, polar_ventricle_b = _get_input(output_dir)

    output_dir = os.path.join(output_dir, "postprocessing")
    os.makedirs(output_dir, exist_ok=True)

    nms_result = get_nms_result(config=config,
                                score_data=score_data,
                                nms_type=config["nms_type"],
                                output_dir=output_dir)
    
    if not nms_result:
        print(f"nms_type: {config['nms_type']} is invalid.")
        return
    
    best_threshold, nms_result = dummy_decider(
        nms_result=nms_result,
        data_dict=data_dict,
        thresholds_dir=thresholds_dir,
        polar_grad=polar_grad,
        polar_ventricle_b=polar_ventricle_b
    )

    with open(os.path.join(output_dir, "nms_result.json"), 'w') as f:
        json.dump(nms_result, f, indent=2)

    return best_threshold