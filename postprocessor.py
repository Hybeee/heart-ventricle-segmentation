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

PRINT = True

def _get_valley_indices(polar_grad: np.ndarray, boundary_points):
    global PRINT
    valley_data = ValleyData(polar_grad=polar_grad)
    result = valley_data.get_valley_data(boundary_points=boundary_points)

    (assigned_valleys, assigned_valley_positions) = result

    # if PRINT:
    #     PRINT = False
    #     fo_grad = valley_data.fo_deriv[:, 5]
    #     boundary_point = boundary_points[5]
        
    #     valley_r = assigned_valley_positions[5]
    #     valley_v = fo_grad[valley_r]

    #     r = boundary_point[0]
    #     v = fo_grad[r]


    #     plt.plot(fo_grad)
    #     plt.scatter(r, v, c='red', s=5, marker='o')
    #     valley_positions = valley_data.valley_positions[5]
    #     for pos in valley_positions:
    #         r = pos
    #         v = fo_grad[r]
    #         plt.scatter(r, v, c='orange', s=30, marker='o')
    #     plt.scatter(valley_r, valley_v, c='purple', s=20, marker='o')
    #     plt.title(f"Valley index: {assigned_valleys[5]}")
    #     plt.show()

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
    pass

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
    
    best_threshold, nms_result = dummy_decider(
        nms_result=nms_result,
        data_dict=input_object.data_dict,
        thresholds_dir=input_object.thresholds_dir,
        polar_grad=input_object.polar_grad,
        polar_ventricle_b=input_object.polar_ventricle_b
    )

    with open(os.path.join(output_dir, "nms_result.json"), 'w') as f:
        json.dump(nms_result, f, indent=2)

    return best_threshold