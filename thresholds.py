import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure

import utils

import os
import json

def _save_threshold_masks(output_dir,
                          mask, mask_boundary,
                          polar_mask, polar_mask_boundary):    
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "mask.npy"), mask)
    np.save(os.path.join(output_dir, "mask_boundary.npy"), mask_boundary)
    np.save(os.path.join(output_dir, "polar_mask.npy"), polar_mask)
    np.save(os.path.join(output_dir, "polar_mask_boundary.npy"), polar_mask_boundary)
    

def _create_threshold_masks(ct, ventricle_mask, polar_converter: utils.PolarConverter,
                           config, output_dir, sigma=1.5):
    output_dir = os.path.join(output_dir, "np")

    ct = gaussian_filter(ct, sigma=sigma)

    threshold_min = np.percentile(ct[ventricle_mask > 0], 5)
    threshold_max = np.percentile(ct[ventricle_mask > 0], 95)

    step = (threshold_max - threshold_min) * config['threshold_step_ratio']

    thresholds = np.arange(threshold_min, threshold_max, step)
    used_thresholds = []

    masks = []
    mask_boundaries = []
    polar_masks = []
    polar_mask_boundaries = []

    for threshold in thresholds:
        mask = np.zeros_like(ct)
        mask[ct > threshold] = 1
        mask[ventricle_mask == 0] = 0
        if np.all(mask == 0):
            print(f"Threshold {threshold} resulted in empty mask. Skipping.")
            continue
        labels = measure.label(mask)
        props = measure.regionprops(labels)
        largest_label = max(props, key=lambda x: x.area).label
        mask = (labels == largest_label).astype(mask.dtype)
        mask_boundary = utils.get_mask_boundary(mask=mask)

        polar_mask = polar_converter.cv2WarpPolar(image=mask.astype(np.int8))
        polar_mask = (polar_mask != 0).astype(polar_mask.dtype)
        polar_mask_boundary = utils.get_polar_boundary_points(polar_mask=polar_mask,
                                                              theta_step_size=config['theta_step_size'])
        
        threshold = round(float(threshold), 2)
        curr_output_dir = os.path.join(output_dir, str(threshold))
        _save_threshold_masks(output_dir=curr_output_dir,
                              mask=mask, mask_boundary=mask_boundary,
                              polar_mask=polar_mask, polar_mask_boundary=polar_mask_boundary)

        masks.append(mask)
        mask_boundaries.append(mask_boundary)
        polar_masks.append(polar_mask)
        polar_mask_boundaries.append(polar_mask_boundary)
        used_thresholds.append(threshold)

    return masks, mask_boundaries, polar_masks, polar_mask_boundaries, used_thresholds

def _get_common_points(polar_ventricle_b, polar_threshold_b,
                       epsilon, data_dict):
    common_count = 0

    lung_data = data_dict["lung_data"]
    starts = lung_data["starts"]
    ends = lung_data["ends"]

    for i in range(len(polar_ventricle_b)):
        in_relevant_region = False
        for start, end in zip(starts, ends):
            if i >= start and i <= end:
                in_relevant_region = True
        
        if not in_relevant_region:
            continue

        r_v = polar_ventricle_b[i][0]
        r_th = polar_threshold_b[i][0]
        
        if abs(r_v - r_th) <= epsilon:
            common_count += 1

    
    return common_count

def _get_mean_radial_difference(polar_threshold_b, data_dict):
    lung_data = data_dict["lung_data"]
    starts = lung_data["starts"]
    ends = lung_data["ends"]
    
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

def _get_positive_gradient_count(polar_grad, polar_threshold_b, data_dict):
    polar_threshold_b = np.array(polar_threshold_b)
    r_indices = polar_threshold_b[:, 0].astype(int)
    theta_indices = polar_threshold_b[:, 1].astype(int)

    lung_data = data_dict["lung_data"]
    starts = lung_data["starts"]
    ends = lung_data["ends"]

    valid_mask = ~((theta_indices >= ends[0]) & (theta_indices <= starts[1]))

    r_indices = r_indices[valid_mask]
    theta_indices = theta_indices[valid_mask]

    values_at_boundary = polar_grad[r_indices, theta_indices]

    return values_at_boundary[values_at_boundary > 0].shape[0]

def _rank_thresholds(polar_grad, polar_ventricle_b,
                     thresholds, polar_threshold_bs,
                     config, data_dict, output_dir):
    score_data = {}

    ALPHA = config["weights"]["alpha"]
    BETA = config["weights"]["beta"]
    GAMMA = config["weights"]["gamma"]

    min_score = np.inf
    best_threshold = None

    for threshold, polar_threshold_b in zip(thresholds, polar_threshold_bs):
        if polar_threshold_b.shape[0] != polar_ventricle_b.shape[0]:
            continue

        common_points = _get_common_points(polar_ventricle_b=polar_ventricle_b,
                                           polar_threshold_b=polar_threshold_b,
                                           epsilon=config["common_points_epsilon"],
                                           data_dict=data_dict)

        mean_radial_difference = _get_mean_radial_difference(polar_threshold_b=polar_threshold_b,
                                                             data_dict=data_dict)

        positive_gradient_count = _get_positive_gradient_count(polar_grad=polar_grad,
                                                               polar_threshold_b=polar_threshold_b,
                                                               data_dict=data_dict)

        score = ALPHA * common_points + BETA * mean_radial_difference + GAMMA * positive_gradient_count

        avg_r = polar_threshold_b[:, 0].mean()

        score_data[threshold] = {
            "common_points": common_points,
            "mean_radial_difference": mean_radial_difference,
            "positive_gradient_count": positive_gradient_count,
            "score": score,
            "avg_r": avg_r
        }

        if score < min_score:
            min_score = score
            best_threshold = threshold

    score_data = dict(
        sorted(score_data.items(), key=lambda item: item[1]["score"])
    )

    print(f"Best Threshold and Score:\n\t{best_threshold}: {min_score}")

    with open(os.path.join(output_dir, "score_data.json"), 'w') as f:
        json.dump(score_data, f, indent=2)

def _get_input(input_dir):
    ct = np.load(os.path.join(input_dir, "np", "ct.npy"))
    polar_grad = np.load(os.path.join(input_dir, "np", "polar_dir_grad.npy"))
    polar_ventricle_b = np.load(os.path.join(input_dir, "np", "polar_ventricle_boundary.npy"))
    ventricle_mask = np.load(os.path.join(input_dir, "np", "ventricle.npy"))
    with open(os.path.join(input_dir, "data.json")) as f:
        data_dict = json.load(f)
    
    return ct, polar_grad, ventricle_mask, polar_ventricle_b, data_dict


def create_and_rank_threshold_masks(config, output_dir, polar_converter):
    preprocess_dir = os.path.join(output_dir, "preprocessing")

    ct, polar_grad, ventricle_mask, polar_ventricle_b, data_dict = _get_input(input_dir=preprocess_dir)

    output_dir = os.path.join(output_dir, "thresholds")
    os.makedirs(output_dir, exist_ok=True)

    masks, mask_boundaries, polar_masks, polar_mask_boundaries, used_thresholds = _create_threshold_masks(ct=ct,
                                                                                                          ventricle_mask=ventricle_mask,
                                                                                                          polar_converter=polar_converter,
                                                                                                          config=config,
                                                                                                          output_dir=output_dir)

    print("Ranking thresholds!")

    _rank_thresholds(
        polar_grad=polar_grad,
        polar_ventricle_b=polar_ventricle_b,
        thresholds=used_thresholds,
        polar_threshold_bs=polar_mask_boundaries,
        config=config,
        data_dict=data_dict,
        output_dir=output_dir
    )