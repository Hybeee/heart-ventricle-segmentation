import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

class StreakingArtifactDetector:
    def __init__(self, func, smooth=True, verbose=False):
        self.func = func
        if not smooth:
            self.func = gaussian_filter1d(func, sigma=2, mode='wrap')

        self.verbose = verbose

        self.peaks, _ = find_peaks(self.func)
        self.valleys, _ = find_peaks(-self.func)

        self.local_window = 100
        self.min_grad_value = 200
        self.max_length = 50

    def return_artifacts(self) -> tuple[int, int, int]:
        peak_heights = self.func[self.peaks]

        z_scores_local = np.zeros(len(self.peaks))
        for i, peak in enumerate(self.peaks):
            dists = np.abs(self.peaks - peak)
            dists = np.minimum(dists, 360 - dists)
            neighborhood = peak_heights[dists <= self.local_window]

            if peak < self.min_grad_value:
                z_scores_local[i] = 0.0
                continue

            if len(neighborhood) < 3:
                z_scores_local[i] = 0.0
                continue

            local_mean = neighborhood.mean()
            local_std = neighborhood.std()

            if local_std == 0:
                z_scores_local[i] = 0.0
                continue

            z_scores_local[i] = (peak_heights[i] - local_mean) / local_std

        artifact_peaks = self.peaks[z_scores_local > 2.0]

        artifacts = []
        for peak in artifact_peaks:
            left_valleys = self.valleys[self.valleys < peak]
            right_valleys = self.valleys[self.valleys > peak]

            if len(left_valleys) == 0 or len(right_valleys) == 0:
                continue

            start = left_valleys[-1]
            end = right_valleys[0]

            if end - start > self.max_length:
                continue

            artifacts.append((start, peak, end))

        return artifacts
        

def _plot_1d_func(values, xlabel, ylabel):
    plt.plot(values)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def _process_data(patient_id, show_plots=False):
        patient_dir = os.path.join("vis_output", patient_id)

        grad_map_path = os.path.join(patient_dir, "preprocessing", "np", "polar_dir_grad.npy")
        grad = np.load(grad_map_path)
        
        ventricle_boundary_path = os.path.join(patient_dir, "preprocessing", "np", "polar_ventricle_boundary.npy")
        ventricle_boundary = np.load(ventricle_boundary_path)

        radii = ventricle_boundary[:, 0].astype(int)
        theta = ventricle_boundary[:, 1].astype(int)

        values = grad[radii, theta]

        smoothed_values = gaussian_filter1d(values, sigma=2, mode='wrap')

        if show_plots:
            plt.imshow(grad, cmap='jet')
            plt.scatter(
                ventricle_boundary[:, 1],
                ventricle_boundary[:, 0],
                s=2,
                c='blue'
            )
            plt.show()
    
        artifact_detector = StreakingArtifactDetector(
            func=smoothed_values,
            smooth=True
        )
        artifacts = artifact_detector.return_artifacts()

        if show_plots:
            _plot_1d_func(smoothed_values, "Theta", "Value")

            deriv = np.diff(smoothed_values, append=smoothed_values[0])

            _plot_1d_func(deriv, "Theta", "Value")

        if artifacts:
            print(patient_id)
        for artifact in artifacts:
            start, peak, end = artifact
            print(f"\tStart: {start}")
            print(f"\tPeak: {peak}")
            print(f"\tEnd: {end}")
            print("======")

def main():
    # patients_to_process = []
    # with open(os.path.join(ROOT_DIR, "patients_to_process.txt"), 'r') as file:
    #     for line in file:
    #         line = line.strip()
    #         patients_to_process.append(line)

    # for patient_id in patients_to_process:
    #     if patient_id == "patient_0025":
    #         continue
    #     _process_data(patient_id)

    patient_id = "patient_0045"
    _process_data(patient_id, show_plots=True)

if __name__ == "__main__":
    main()