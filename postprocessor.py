import numpy as np

import os
import json

class InputObject:
    def __init__(self, input_dir):
        with open(os.path.join(input_dir, "preprocessing", "data.json"), 'r') as f:
            self.data_dict = json.load(f)

        self.thresholds_dir = os.path.join(input_dir, "thresholds", "np")

        self.polar_grad = np.load(os.path.join(input_dir, "preprocessing", "np", "polar_dir_grad.npy"))

        self.ventricle_mask = np.load(os.path.join(input_dir, "preprocessing", "np", "ventricle.npy"))
        self.polar_ventricle_b = np.load(os.path.join(input_dir, "preprocessing", "np", "polar_ventricle_boundary.npy"))

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

class PostProcessor():
    def __init__(self, config, output_dir):
        self.input_object = InputObject(input_dir=output_dir)
        self.config = config

        output_dir = os.path.join(output_dir, "postprocessing")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
    
    def _get_validity(self, threshold_area, ventricle_area):
        area_ratio = threshold_area / ventricle_area

        valid = ((area_ratio > self.config["area_ratio_lower_threshold"])
                * (area_ratio < self.config["area_ratio_upper_threshold"]))
        
        return valid

    def _calculate_valley_score(self,
                                polar_threshold_b,
                                valley_data: ValleyData,
                                starts: list[int], ends: list[int],
                                threshold: str):
        r_indices = polar_threshold_b[:, 0].astype(int)
        theta_indices = polar_threshold_b[:, 1].astype(int)

        t_valley_data = valley_data.get_valley_data(boundary_points=polar_threshold_b)
        (t_assigned_valleys, t_assigned_valley_positions) = t_valley_data

        v_valley_data = valley_data.get_valley_data(boundary_points=self.input_object.polar_ventricle_b)
        (v_assigned_valleys, _) = v_valley_data

        # valid_mask = (
        #     ~((theta_indices >= ends[0]) & (theta_indices <= starts[1]))
        # ) & (r_indices != -1) & (t_assigned_valleys != v_assigned_valleys)

        ventricle_r_indices = self.input_object.polar_ventricle_b[:, 0].astype(int)
        ventricle_theta_indices = self.input_object.polar_ventricle_b[:, 1].astype(int)
        ventricle_values = self.input_object.polar_grad[ventricle_r_indices, ventricle_theta_indices]

        valid_mask = ((r_indices > 0) & 
                    ((t_assigned_valleys != v_assigned_valleys) | (ventricle_values > 0)))

        r_valid = r_indices[valid_mask]
        theta_valid = theta_indices[valid_mask]
        valley_r_valid = t_assigned_valley_positions[valid_mask]

        weights = np.array([abs(self.input_object.polar_grad[int(rv), theta])
                            for rv, theta in zip(valley_r_valid, theta_valid)])
        distances = (r_valid - valley_r_valid)**2

        weights_sum = np.sum(weights)
        score = np.sum(weights * distances) / weights_sum if weights_sum > 0 else np.inf

        valid_indices = np.where(valid_mask)[0]

        score_data = {
            "length": len(valid_indices),
            "mean_valley": float(t_assigned_valleys.mean()),
            "var_valley": float(t_assigned_valleys.var()),
            "weights_sum": float(weights_sum) if weights_sum > 0 else np.inf,
            "bp_data": "No detailed bp/valley data."
        }

        if self.config["include_detailed_valley_data"]:
            score_data["bp_data"] = [
                    (int(index), float(weight), float(distance))
                    for index, weight, distance in zip(valid_indices, weights, distances) 
            ]
            
        return score, score_data

    def _get_mean_radial_difference(self,
                                    polar_threshold_b: np.ndarray,
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

    def _normalize_results(result_dict):
        valley_scores = []
        mrd_scores = []

        for _, v in result_dict.items():
            if not v["valid"]:
                continue

            valley_scores.append(v["score_components"]["valley_score"])
            mrd_scores.append(v["score_components"]["mean_radial_difference"])
        
        mean_valley = np.array(valley_scores).mean()
        var_valley = np.array(valley_scores).std()

        mean_mrd = np.array(mrd_scores).mean()
        var_mrd = np.array(mrd_scores).std()

        new_dict = {}

        for k, v in result_dict.items():
            valley_score = v["score_components"]["valley_score"]
            mrd_score = v["score_components"]["mean_radial_difference"]

            norm_valley = (valley_score - mean_valley) / var_valley
            norm_mrd = (mrd_score - mean_mrd) / var_mrd

            new_dict[k] = {
                "valid": v["valid"],
                "total_score": float(norm_valley + norm_mrd),
                "score_components": {
                    "valley_score": 2.0 * float(norm_valley),
                    "mean_radial_difference": float(norm_mrd)
                },
                "valley_score_data": v["valley_score_data"]
            }

        return new_dict

    def _approximate_best_threshold(self):
        
        result_dict = {}

        best_threshold = None
        best_score = np.inf

        lung_data = self.input_object.data_dict["lung_data"]
        starts = lung_data["starts"]
        ends = lung_data["ends"]

        valley_data = ValleyData(polar_grad=self.input_object.polar_grad)

        thresholds = os.listdir(self.input_object.thresholds_dir)

        ventricle_area = np.sum(self.input_object.ventricle_mask == 1)

        for threshold in thresholds:
            threshold_dir = os.path.join(self.input_object.thresholds_dir, threshold)
            threshold_mask = np.load(os.path.join(threshold_dir, "mask.npy"))
            threshold_area = np.sum(threshold_mask == 1)

            valid = self._get_validity(threshold_area=threshold_area,
                                       ventricle_area=ventricle_area)

            polar_threshold_b = np.load(os.path.join(threshold_dir, "polar_mask_boundary.npy"))

            valley_score, valley_score_data = self._calculate_valley_score(
                polar_threshold_b=polar_threshold_b,
                valley_data=valley_data,
                starts=starts,
                ends=ends,
                threshold=threshold
            )

            mean_radial_difference = self._get_mean_radial_difference(
                polar_threshold_b=polar_threshold_b,
                starts=starts,
                ends=ends
            )

            score = valley_score + mean_radial_difference
            if score < best_score and valid:
                best_score = score
                best_threshold = threshold

            if valid:
                result_dict[threshold] = {
                    "total_score": float(score),
                    "score_components": {
                        "valley_score": float(valley_score),
                        "mean_radial_difference": float(mean_radial_difference)
                    },
                    "valley_score_data": valley_score_data
                }

        # NOTE: minmax norm probably won't be good
        # result_dict = _normalize_results(result_dict)

        result_dict = dict(
            sorted(result_dict.items(), key=lambda item: item[1]["total_score"])
        )

        best_threshold = None
        for threshold in result_dict.keys():
            best_threshold = threshold
            break

        return best_threshold, result_dict

    def get_3d_mask(self, ct, best_threshold):
        assert ct.shape == self.input_object.ventricle_mask.shape and len(ct.shape) == 3

        best_threshold = float(best_threshold)

        mask = np.zeros_like(ct)
        mask[ct > best_threshold] = 1
        mask[self.input_object.ventricle_mask == 0] = 0

        return mask

    def calculate_approximation(self):
        best_threshold, result_dict = self._approximate_best_threshold()

        with open(os.path.join(self.output_dir, "results.json"), 'w') as f:
            json.dump(result_dict, f, indent=2)

        return best_threshold
