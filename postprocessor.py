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

    for threshold in nms_result.keys():
        curr_result = nms_result[threshold]
        curr_result["valid"] = ((curr_result["area_ratio"] > config["area_ratio_lower_threshold"])
                                & (curr_result["area_ratio"] < config["area_ratio_upper_threshold"]))

    if nms_result:
        with open(os.path.join(output_dir, "nms_result.json"), 'w') as f:
            json.dump(nms_result, f, indent=2)
    
    return nms_result

class ValleyData:
    def __init__(self, polar_grad: np.ndarray):
        self.fo_deriv = polar_grad
        self.so_deriv = np.gradient(self.fo_deriv, axis=0)

        self.zero_crossings = np.diff(np.signbit(self.so_deriv), axis=0)

        self.valley_slope = self.so_deriv[1:, :] - self.so_deriv[:-1, :]
        self.valley_mask = self.zero_crossings & (self.valley_slope > 0)
        self.peak_slope = -self.so_deriv[1:, :] + self.so_deriv[:-1, :]
        self.peak_mask = self.zero_crossings & (self.peak_slope > 0)

        self.valley_positions = [np.where(self.valley_mask[:, c])[0] for c in range(polar_grad.shape[1])]
        self.peak_positions = [np.where(self.peak_mask[:, c])[0] for c in range(polar_grad.shape[1])]

    def get_valley_data(self, boundary_points: np.ndarray):
        r_idx, theta_idx = boundary_points[:, 0], boundary_points[:, 1]
        
        assigned_valleys = np.full(len(r_idx), -1, dtype=int)
        assigned_valley_positions = np.full(len(r_idx), -1, dtype=int)

        for i, (r, t) in enumerate(zip(r_idx, theta_idx)):
            valley_position = self.valley_positions[t]
            peak_position = self.peak_positions[t]

            if len(valley_position) == 0:
                assigned_valleys[i] = -1
                assigned_valley_positions[i] = -1
                continue

            valley_idx = np.searchsorted(valley_position, r, side='right') - 1
            if valley_idx < 0:
                assigned_valleys[i] = -1
                assigned_valley_positions[i] = -1
                continue
            
            next_valley = valley_position[valley_idx + 1] if (valley_idx + 1) < len(valley_position) else self.fo_deriv.shape[0]
            peaks_between = peak_position[(peak_position > valley_position[valley_idx]) & (peak_position < next_valley)]

            if len(peaks_between) > 0 and r > peaks_between[0]:
                valley_idx += 1
                valley_idx = min(valley_idx, len(valley_position)-1)
            
            assigned_valleys[i] = valley_idx
            assigned_valley_positions[i] = self.valley_positions[t][valley_idx]
        
        return (assigned_valleys, assigned_valley_positions)

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

        # assigned_valleys_t = _get_valley_indices(
        #     polar_grad=polar_grad,
        #     boundary_points=polar_threshold_b)
        # assigned_valleys_v = _get_valley_indices(
        #     polar_grad=polar_grad,
        #     boundary_points=polar_ventricle_b
        # )

        # if np.any(assigned_valleys_t == assigned_valleys_v):
        #     print(f"Skipping {threshold}.")
        #     continue

        print(f"\t{threshold}: {total}")
        nms_result[threshold]["total_depth"] = float(total)

        if total > best_score:
            best_score = total
            best_threshold = threshold

    return best_threshold, nms_result

def distance_decider(nms_result: dict, data_dict: dict, thresholds_dir: str,
                     polar_grad: np.ndarray, polar_ventricle_b: np.ndarray):
    best_threshold = None
    best_score = np.inf

    lung_data = data_dict["lung_data"]
    starts = lung_data["starts"]
    ends = lung_data["ends"]

    valley_data = ValleyData(polar_grad=polar_grad)

    for threshold in nms_result.keys():
        threshold_dir = os.path.join(thresholds_dir, threshold)
        polar_threshold_b = np.load(os.path.join(threshold_dir, "polar_mask_boundary.npy"))
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

        score = np.sum(weights * distances) / np.sum(weights)
        nms_result[threshold]["distance_decider_score"] = float(score)
        
        valid = nms_result[threshold]["valid"]
        if score < best_score and valid:
            best_score = score
            best_threshold = threshold

    return best_threshold, nms_result


class InputObject:
    def __init__(self, input_dir):
        with open(os.path.join(input_dir, "thresholds", "score_data.json"), 'r') as f:
            self.score_data = json.load(f)

        with open(os.path.join(input_dir, "preprocessing", "data.json"), 'r') as f:
            self.data_dict = json.load(f)

        self.thresholds_dir = os.path.join(input_dir, "thresholds", "np")

        self.polar_grad = np.load(os.path.join(input_dir, "preprocessing", "np", "polar_dir_grad.npy"))

        self.polar_ventricle_b = np.load(os.path.join(input_dir, "preprocessing", "np", "polar_ventricle_boundary.npy"))

def _get_input(input_dir) -> InputObject:
    return InputObject(input_dir=input_dir)

def postprocess(config, output_dir):

    input_object = _get_input(output_dir)

    output_dir = os.path.join(output_dir, "postprocessing")
    os.makedirs(output_dir, exist_ok=True)

    nms_result = get_nms_result(config=config,
                                score_data=input_object.score_data,
                                nms_type=config["nms_type"],
                                output_dir=output_dir)
    
    if not nms_result:
        print(f"nms_type: {config['nms_type']} is invalid.")
        return
    
    best_threshold, nms_result = distance_decider(
        nms_result=nms_result,
        data_dict=input_object.data_dict,
        thresholds_dir=input_object.thresholds_dir,
        polar_grad=input_object.polar_grad,
        polar_ventricle_b=input_object.polar_ventricle_b
    )

    with open(os.path.join(output_dir, "nms_result.json"), 'w') as f:
        json.dump(nms_result, f, indent=2)

    return best_threshold