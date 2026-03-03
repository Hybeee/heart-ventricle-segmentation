import numpy as np

import os
import json

from postprocessor import ValleyData

def _get_validity(config, threshold_area, ventricle_area):
    area_ratio = threshold_area / ventricle_area

    valid = ((area_ratio > config["area_ratio_lower_threshold"])
             * (area_ratio < config["area_ratio_upper_threshold"]))
    
    return valid

def _calculate_valley_score(polar_grad, polar_threshold_b, polar_ventricle_b,
                            valley_data: ValleyData,
                            starts: list[int], ends: list[int]):
    r_indices = polar_threshold_b[:, 0].astype(int)
    theta_indices = polar_threshold_b[:, 1].astype(int)

    t_valley_data = valley_data.get_valley_data(boundary_points=polar_threshold_b)
    (t_assigned_valleys, t_assigned_valley_positions) = t_valley_data

    v_valley_data = valley_data.get_valley_data(boundary_points=polar_ventricle_b)
    (v_assigned_valleys, _) = v_valley_data

    valid_mask = (
        ~((theta_indices >= ends[0]) & (theta_indices <= starts[1]))
    ) & (r_indices != -1) & (t_assigned_valleys != v_assigned_valleys)

    r_valid = r_indices[valid_mask]
    theta_valid = theta_indices[valid_mask]
    valley_r_valid = t_assigned_valley_positions[valid_mask]

    weights = np.array([abs(polar_grad[int(rv), theta])
                        for rv, theta in zip(valley_r_valid, theta_valid)])
    distances = (r_valid - valley_r_valid)**2

    weights_sum = np.sum(weights)
    score = np.sum(weights * distances) / weights_sum if weights_sum > 0 else np.inf

    score_data = {
        "valid_mask_data": {
            "length": len(np.where(valid_mask)[0]),
            "valid_mask_indices": np.where(valid_mask)[0].tolist()
        },
        "distances^2": distances.tolist(),
        "valley_data": {
            "mean_valley": float(t_assigned_valleys.mean()),
            "var_valley": float(t_assigned_valleys.var()),
            "assigned_valleys": t_assigned_valleys.tolist()
        }
    }

    return score, score_data

def _get_mean_radial_difference(polar_threshold_b: np.ndarray,
                                starts: list[int], ends: list[int]):
    radial_difference = []

    for i in range(len(polar_threshold_b) - 1):
        in_relevant_region = False
        for start, end in zip(starts, ends):
            if i >= start and i <= end:
                in_relevant_region = True
        
        if not in_relevant_region:
            continue

        r1 = polar_threshold_b[i][0]
        r2 = polar_threshold_b[i+1][0]

        radial_difference.append(abs(r2-r1))
    
    return np.array(radial_difference).mean()

def _approximate_best_threshold(config, ventricle_mask,
                                polar_grad, polar_ventricle_b,
                                thresholds_dir: str, data_dict: dict):
    
    result_dict = {}

    best_threshold = None
    best_score = np.inf

    lung_data = data_dict["lung_data"]
    starts = lung_data["starts"]
    ends = lung_data["ends"]

    valley_data = ValleyData(polar_grad=polar_grad)

    thresholds = os.listdir(thresholds_dir)

    ventricle_area = np.sum(ventricle_mask == 1)

    for threshold in thresholds:
        threshold_dir = os.path.join(thresholds_dir, threshold)
        threshold_mask = np.load(os.path.join(threshold_dir, "mask.npy"))
        threshold_area = np.sum(threshold_mask == 1)

        valid = _get_validity(config=config,
                              threshold_area=threshold_area,
                              ventricle_area=ventricle_area)

        polar_threshold_b = np.load(os.path.join(threshold_dir, "polar_mask_boundary.npy"))

        valley_score, valley_score_data = _calculate_valley_score(
            polar_grad=polar_grad,
            polar_threshold_b=polar_threshold_b,
            polar_ventricle_b=polar_ventricle_b,
            valley_data=valley_data,
            starts=starts,
            ends=ends
        )

        mean_radial_difference = _get_mean_radial_difference(
            polar_threshold_b=polar_threshold_b,
            starts=starts,
            ends=ends
        )

        score = valley_score + mean_radial_difference
        if score < best_score and valid:
            best_score = score
            best_threshold = threshold

        result_dict[threshold] = {
            "valid": bool(valid),
            "total_score": float(score),
            "score_components": {
                "valley_score": float(valley_score),
                "mean_radial_difference": float(mean_radial_difference)
            },
            "valley_score_data": valley_score_data
        }

    result_dict = dict(
        sorted(result_dict.items(), key=lambda item: item[1]["total_score"])
    )

    return best_threshold, result_dict

class InputObject:
    def __init__(self, input_dir):
        with open(os.path.join(input_dir, "preprocessing", "data.json"), 'r') as f:
            self.data_dict = json.load(f)

        self.thresholds_dir = os.path.join(input_dir, "thresholds", "np")

        self.polar_grad = np.load(os.path.join(input_dir, "preprocessing", "np", "polar_dir_grad.npy"))

        self.ventricle_mask = np.load(os.path.join(input_dir, "preprocessing", "np", "ventricle.npy"))
        self.polar_ventricle_b = np.load(os.path.join(input_dir, "preprocessing", "np", "polar_ventricle_boundary.npy"))

def _get_input(input_dir) -> InputObject:
    return InputObject(input_dir=input_dir)

def calculate_approximation(config, output_dir):
    input_object = _get_input(input_dir=output_dir)

    output_dir = os.path.join(output_dir, "postprocessing")
    os.makedirs(output_dir, exist_ok=True)

    best_threshold, result_dict = _approximate_best_threshold(
        config=config,
        ventricle_mask=input_object.ventricle_mask,
        polar_grad=input_object.polar_grad,
        polar_ventricle_b=input_object.polar_ventricle_b,
        thresholds_dir=input_object.thresholds_dir,
        data_dict=input_object.data_dict
    )

    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(result_dict, f, indent=2)

    return best_threshold
