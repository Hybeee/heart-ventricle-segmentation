import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

import time
import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

GAC_PARAMS = {
    "curvature_scaling": 1.00,
    "advection_scaling": 1.00,
    "propagation_scaling": 0.2976,
    "num_iterations": 1000,
    "max_rms_error": 0.00001
}

GAC_CHECKPOINTS = [10, 20, 30, 40, 50, 60, 70, 80, 90,
        100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
        225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500,
        550, 600, 650, 700, 750, 800, 850, 900, 950, 1000,
        1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500,
        3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000,
        6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000, 8250, 8500,
        8750, 9000, 9250, 9500, 9750, 10000]

ADAPTIVE_PARAMS = {
    "median_window": 5,
    "gamma": 0.90,
    "delta": 1.10,
    "stall_window": 20, # dI * stall_window = "patience"
    "warmup_iter": 100,
    "max_distance_mm": 17,
    "converged_max_val": 1e-2, # value should be smaller than this,
    "rollback_dist": 5 # mm
}

class ProgressMonitor():
    def __init__(self, dist_traf_orig, roi_mins, mask, prev_binary_mask,
                 exp_name):

        # convergence detection data
        self.rms_changes = []
        self.boundary_changes = []
        self.smoothed_bc = []
        self.iters = []
        self.flipped_voxels = []
        self.ap_window = ADAPTIVE_PARAMS["median_window"]
        self.has_conv = False
        self.prev_binary_mask = prev_binary_mask

        # bubble detection data
        self.dist_traf_orig = dist_traf_orig
        self.max_distances = []
        self.n_mm = ADAPTIVE_PARAMS["max_distance_mm"]
        self.current_far = 0
        self.current_touches_roi = False
        self.current_far_mask = None
        self.balloon_det = False

        # general data
        self.roi_mins = roi_mins
        self.mask = mask
        self.current_iteration = None # total iterations ran so far.
        self.exp_name = exp_name
        self.runs = {}

    def update(self, iter_data: dict):
        self.iters.append(iter_data["elapsed_iter"]) # how long GAC ran the last time for
        self.rms_changes.append(iter_data["rms_change"])
        self.current_iteration = iter_data["current_iteration"]

        result_binary_mask = iter_data["result_binary_mask"]
        boundary_displacement, flipped = get_boundary_displacement(
            A=result_binary_mask,
            B=self.prev_binary_mask
        )
        self.boundary_changes.append(boundary_displacement)
        self.flipped_voxels.append(flipped)
        if len(self.boundary_changes) >= self.ap_window:
            self.smoothed_bc.append(np.median(self.boundary_changes[-self.ap_window:]))
        self.prev_binary_mask = result_binary_mask

        far_mask = result_binary_mask & (self.dist_traf_orig > self.n_mm)
        far = int(np.count_nonzero(far_mask))
        self.current_far_mask = far_mask
        self.current_far = far
        self.current_touches_roi = touches_roi_wall(result_binary_mask)
        max_distance = float(self.dist_traf_orig[result_binary_mask].max())
        self.max_distances.append((self.current_iteration, max_distance))

    def has_converged(self):
        self.has_conv = is_converged(self.smoothed_bc, self.current_iteration)

        message = "" if not self.has_conv else f"\tConverged at iteration: {self.current_iteration}!"

        return {
            "has_conv": self.has_conv,
            "message": message,
        }

    def detect_bubble(self):
        self.balloon_det = self.current_far > 0 or self.current_touches_roi

        if not self.balloon_det:
            return {
                "bubble_det": False,
                "message": "",
                "rollback_iteration": None
            }

        example_zyx_message = ""

        if self.current_far > 0:
            example_zyx = np.argwhere(self.current_far_mask)[0]
            z_min, y_min, x_min = self.roi_mins
            full_zyx = example_zyx + np.array([z_min, y_min, x_min])
            z, y, x = tuple(int(v) for v in full_zyx)
            z = self.mask.shape[0] - z
            example_zyx_message = f"\n\tExample voxel (z, y, x): ({z}, {y}, {x})"

        below = [(it, md) for it, md in self.max_distances if md < ADAPTIVE_PARAMS["rollback_dist"]]
        it_best, md_best = max(below, key=lambda t: t[1]) if below else (None, None)

        message = (
            f"\tBubble detected at iteration: {self.current_iteration}" +
            f"\n\tTouches ROI wall: {self.current_touches_roi}" +
            f"\n\tProposed rollback value: {it_best} - distance: {md_best}" +
            f"{example_zyx_message}"
        )

        return {
            "bubble_det": self.balloon_det,
            "message": message,
            "rollback_data": (it_best, md_best)
        }

    def _full(self, first, n, key, current):
        return np.concatenate([np.asarray(first[key][:n], dtype=float),
                               np.asarray(current, dtype=float)])

    def _snapshot(self):
        return {
            "rms_changes": self.rms_changes,
            "boundary_changes": self.boundary_changes,
            "smoothed_bc": self.smoothed_bc,
            "iters": self.iters,
            "flipped_voxels": self.flipped_voxels,
            "max_distances": self.max_distances
        }

    def save_results(self, first_exp_name, n, output_dir):
        runs = {**self.runs, self.exp_name: self._snapshot()}
        
        first = runs[first_exp_name]

        for exp_name, items in runs.items():
            iters = items["iters"]
            rms_changes = items["rms_changes"]
            boundary_changes = items["boundary_changes"]
            flipped_voxels = items["flipped_voxels"]

            if exp_name != first_exp_name:
                iters = self._full(first, n, "iters", iters)
                rms_changes = self._full(first, n, "rms_changes", rms_changes)
                boundary_changes = self._full(first, n, "boundary_changes", boundary_changes)
                flipped_voxels = self._full(first, n, "flipped_voxels", flipped_voxels)

            plot_convergence(
                exp_name,
                iters,
                rms_changes,
                boundary_changes,
                flipped_voxels,
                output_dir
            )

    def reset(self, prev_binary_mask, exp_name):
        prev_exp_name = self.exp_name
        runs_history = self.runs
        runs_history[prev_exp_name] = self._snapshot()

        self.__init__(
            self.dist_traf_orig, self.roi_mins, self.mask,
            prev_binary_mask, exp_name
        )
        self.runs = runs_history

def _fmt_param(value: float, decimals: int = 4):
    sign = "n" if value < 0 else ""
    return f"{sign}{abs(value):.{decimals}f}".replace(".", "p")

def _make_mask_name(curv, prop, adv=None):
    parts = [f"c{_fmt_param(curv)}", f"p{_fmt_param(prop)}"]

    if adv is not None:
        parts.append(f"a{_fmt_param(adv)}")

    return "_".join(parts)

def _get_bbox_with_padding(mask_np, padding_voxels):
    coords = np.argwhere(mask_np)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1

    pad = np.asarray(padding_voxels)
    mins = np.maximum(mins - pad, 0)
    maxs = np.minimum(maxs + pad, np.array(mask_np.shape))

    return mins, maxs

def _get_signed_distance(mask_sitk, mins, maxs):
    z_min, y_min, x_min = mins
    z_max, y_max, x_max = maxs

    roi_index = [int(x_min), int(y_min), int(z_min)]
    roi_size = [int(x_max - x_min), int(y_max - y_min), int(z_max - z_min)]

    mask_sitk_roi = sitk.RegionOfInterest(mask_sitk, roi_size, roi_index)

    signed_distance = sitk.SignedMaurerDistanceMap(
        mask_sitk_roi,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True
    )
    signed_distance_np = sitk.GetArrayFromImage(signed_distance)

    return signed_distance, signed_distance_np

def _get_gac_inputs(mask_sitk, mask):
    padding_mm = 5.0
    spacing_xyz = mask_sitk.GetSpacing()
    padding_voxels = [
        int(np.ceil(padding_mm / spacing_xyz[2])),
        int(np.ceil(padding_mm / spacing_xyz[1])),
        int(np.ceil(padding_mm / spacing_xyz[0]))
    ]
    mins, maxs = _get_bbox_with_padding(
        mask_np=mask, padding_voxels=padding_voxels
    )

    signed_distance, signed_distance_np = _get_signed_distance(
        mask_sitk=mask_sitk,
        mins=mins,
        maxs=maxs
    )

    initial_level_set = sitk.Cast(signed_distance, sitk.sitkFloat32)

    floor = 0.1
    edge_potential_np = floor + (1 - floor) * (1.0 - np.exp(-(signed_distance_np ** 2) / (2 * 2.0 ** 2)))
    edge_potential = sitk.GetImageFromArray(edge_potential_np.astype(np.float32))
    edge_potential.CopyInformation(initial_level_set)

    return {
        "roi_mins": mins,
        "roi_maxs": maxs,
        "ils": initial_level_set,
        "ep": edge_potential
    }

def surface_voxels(mask):
    return mask & ~ndimage.binary_erosion(mask, border_value=1)

def get_boundary_displacement(A, B):
    flipped = np.count_nonzero(A ^ B)
    surf = np.count_nonzero(surface_voxels(A))

    return flipped / max(surf, 1), flipped

def _level_set_to_binary_mask(level_set):
    current_level_set_np = sitk.GetArrayFromImage(level_set)
    binary_mask = (current_level_set_np < 0).astype(bool)

    return binary_mask

def _get_final_binary_mask(roi_binary_mask, orig_mask, mins, maxs):
    z_min, y_min, x_min = mins
    z_max, y_max, x_max = maxs

    final_binary_mask = np.zeros_like(orig_mask, np.uint8)
    final_binary_mask[z_min:z_max, y_min:y_max, x_min:x_max] = roi_binary_mask

    return final_binary_mask

def causal_median(x, w):
    out = np.full(len(x), np.nan)
    out[w - 1:] = np.median(np.lib.stride_tricks.sliding_window_view(x, w), axis=1)
    return out

def plot_convergence(exp_name, iterations, rms_changes, boundary_changes, flipped_voxels, out_dir):
    series = [
        ["rms_change", rms_changes, "RMS change"],
        ["boundary_displacement", boundary_changes, "Changed voxels/it"],
        ["flipped_voxels", flipped_voxels, "Flipped voxels for boundary displacement"]
    ]

    for name, values, ylabel in series:
        values = np.asarray(values, dtype=float)
        smoothed = causal_median(values, 5)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(iterations, values, color='gray', marker='o', markersize=3,
                linewidth=3, label='Raw')
        ax.plot(iterations, smoothed, color='tab:red', linewidth=2,
                label=f"Rolling median (window=5)")
        ax.set_yscale("log")
        ax.set_xlabel("GAC iterations")
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(os.path.join(out_dir, f"{exp_name}_{name}.png"))
        plt.close(fig)

    remaining = np.cumsum(boundary_changes[::-1])[::-1]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        iterations, remaining,
        marker='o', markersize=3,
        linewidth=2
    )
    ax.set_yscale("log")
    ax.set_xlabel("GAC iterations")
    ax.set_ylabel(f"Remaining boundary displacement until iteration {iterations[-1]}")
    ax.set_title("Cumulative remaining boundary motion")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, f"{exp_name}_remaining_motion.png"))
    plt.close(fig)

def is_converged(smoothed_bc, iteration):
    ap_stall_window = ADAPTIVE_PARAMS["stall_window"]
    gamma = ADAPTIVE_PARAMS["gamma"]
    delta = ADAPTIVE_PARAMS["delta"]
    warmup_iter = ADAPTIVE_PARAMS["warmup_iter"]
    converged_max_val = ADAPTIVE_PARAMS["converged_max_val"]

    if iteration < warmup_iter or len(smoothed_bc) <= ap_stall_window:
        return False

    ratio = smoothed_bc[-1] / max(smoothed_bc[-1 - ap_stall_window], 1e-12)

    in_converged_range = np.all(np.array(smoothed_bc[-ap_stall_window:]) < converged_max_val)

    return gamma < ratio < delta and in_converged_range

def touches_roi_wall(mask):
    return (mask[0].any() or mask[-1].any() or
            mask[:, 0].any() or mask[:, -1].any() or
            mask[:, :, 0].any() or mask[:, :, -1].any())

def _process_patient(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    ct = utils.scan_to_np_array(scan_path=os.path.join(data_dir, "ct.nii.gz"))
    mask_sitk, mask = utils.scan_to_np_array(scan_path=os.path.join(data_dir, "final_mask_nip.seg.nrrd"), return_sitk=True)

    params = GAC_PARAMS.copy()
    for k, v in params.items():
        print(f"{k}: {v}")

    gac_inputs = _get_gac_inputs(mask_sitk, mask)

    initial_level_set = gac_inputs["ils"]
    edge_potential = gac_inputs["ep"]

    iterations = {i: it for i, it in enumerate(range(50, 10050, 50))}
    iter_to_idx = {it: i for i, it in iterations.items()}

    elapsed_iter = 0
    current_level_set = initial_level_set

    exp_name = _make_mask_name(
        curv=params["curvature_scaling"],
        prop=params["propagation_scaling"],
        adv=params["advection_scaling"]
    )
    rollback_it = None
    rollback_exp_name = exp_name

    dist_traf_orig = np.maximum(sitk.GetArrayFromImage(initial_level_set), 0)
    mins = gac_inputs["roi_mins"]
    maxs = gac_inputs["roi_maxs"]

    progress_monitor = ProgressMonitor(
        dist_traf_orig=dist_traf_orig,
        roi_mins=mins,
        mask=mask,
        prev_binary_mask=_level_set_to_binary_mask(initial_level_set),
        exp_name=exp_name
    )

    start = time.time()
    i = 0
    while i < len(iterations.keys()): 
        iteration = iterations[i]
        if iteration % 1000 == 0:
            print(f"\tAt iteration: {iteration}")

        curr_output_dir = os.path.join(out_dir, f"it{iteration}")
        os.makedirs(curr_output_dir, exist_ok=True)

        current_iter = iteration - elapsed_iter
        active_contour = sitk.GeodesicActiveContourLevelSetImageFilter()
        active_contour.SetCurvatureScaling(params["curvature_scaling"])
        active_contour.SetAdvectionScaling(params["advection_scaling"])
        active_contour.SetPropagationScaling(params["propagation_scaling"])
        active_contour.SetMaximumRMSError(params["max_rms_error"])
        active_contour.SetNumberOfIterations(int(current_iter))

        result_level_set = active_contour.Execute(current_level_set, edge_potential)
        result_binary_mask = _level_set_to_binary_mask(result_level_set)
        elapsed_iter += active_contour.GetElapsedIterations()
        
        progress_monitor.update({
            "elapsed_iter": elapsed_iter,
            "rms_change": active_contour.GetRMSChange(),
            "current_iteration": iteration,
            "result_binary_mask": result_binary_mask
        })

        current_level_set = result_level_set
        current_binary_mask = _get_final_binary_mask(
            roi_binary_mask=result_binary_mask,
            orig_mask=mask,
            mins=mins,
            maxs=maxs
        )

        utils.save_data(
            data=current_binary_mask,
            ref_sitk=mask_sitk,
            output_dir=curr_output_dir,
            name=exp_name,
            is_mask=True,
            color="0.1 0.45 0.8",
            segment_name=exp_name
        )

        bubble_det_results = progress_monitor.detect_bubble()
        if bubble_det_results["bubble_det"]:
            print(bubble_det_results["message"])

            if rollback_it is None:
                rollback_it, _ = bubble_det_results["rollback_data"]
                rollback_it = rollback_it - 50 * ADAPTIVE_PARAMS["median_window"] # dI * window size kind of a warmup stage too
                rollback_it = max(rollback_it, 50)
            
            i = iter_to_idx[rollback_it] + 1
            elapsed_iter = rollback_it

            print(f"\tPerforming rollback to {rollback_it} of exp {rollback_exp_name}")

            old_mask_sitk, _ = utils.scan_to_np_array(scan_path=os.path.join(
                out_dir, f"it{rollback_it}", f"{rollback_exp_name}.seg.nrrd"
            ), return_sitk=True)
            old_signed_distance, _ = _get_signed_distance(
                old_mask_sitk,
                mins=mins,
                maxs=maxs
            )
            current_level_set = sitk.Cast(old_signed_distance, sitk.sitkFloat32)

            params["curvature_scaling"] *= 1.5
            exp_name = _make_mask_name(
                curv=params["curvature_scaling"],
                prop=params["propagation_scaling"],
                adv=params["advection_scaling"]
            )
            progress_monitor.reset(
                prev_binary_mask = _level_set_to_binary_mask(current_level_set),
                exp_name=exp_name
            )

            print("\tNew params:")
            for k, v in params.items():
                print(f"\t\t{k}: {v:.2f}")

            continue

        conv_results = progress_monitor.has_converged()
        if conv_results["has_conv"]:
            print(conv_results["message"])
            break

        i += 1

    end = time.time()

    print(f"\tRun took {(end - start):.4f}s")

    results_output_dir = os.path.join(out_dir, "_results")
    os.makedirs(results_output_dir, exist_ok=True)
    progress_monitor.save_results(
        first_exp_name=rollback_exp_name,
        n=(iter_to_idx[rollback_it] + 1) if rollback_it is not None else 0,
        output_dir=results_output_dir
    )

def main():
    data_dir = os.path.join(ROOT_DIR, "pipeline_output")
    out_dir = os.path.join(ROOT_DIR, "adaptive_gac_output_test")
    os.makedirs(out_dir, exist_ok=True)

    for patient_id in sorted(os.listdir(data_dir)):        
        print(f"Processing {patient_id}...")

        _process_patient(
            data_dir=os.path.join(data_dir, patient_id),
            out_dir=os.path.join(out_dir, patient_id),
        )

if __name__ == "__main__":
    main()