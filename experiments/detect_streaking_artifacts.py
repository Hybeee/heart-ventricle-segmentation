import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def _plot_1d_func(values, xlabel, ylabel):
    plt.plot(values)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def _process_data(patient_id):
        print(patient_id)
        patient_dir = os.path.join("vis_output", patient_id)

        grad_map_path = os.path.join(patient_dir, "preprocessing", "np", "polar_dir_grad.npy")
        grad = np.load(grad_map_path)
        
        ventricle_boundary_path = os.path.join(patient_dir, "preprocessing", "np", "polar_ventricle_boundary.npy")
        ventricle_boundary = np.load(ventricle_boundary_path)

        radii = ventricle_boundary[:, 0].astype(int)
        theta = ventricle_boundary[:, 1].astype(int)

        values = grad[radii, theta]

        smoothed_values = gaussian_filter1d(values, sigma=2, mode='wrap')

        # plt.imshow(grad, cmap='jet')
        # plt.scatter(
        #     ventricle_boundary[:, 1],
        #     ventricle_boundary[:, 0],
        #     s=2,
        #     c='blue'
        # )
        # plt.show()

        _plot_1d_func(smoothed_values, "Theta", "Value")

        deriv = np.diff(smoothed_values, append=smoothed_values[0])

        _plot_1d_func(deriv, "Theta", "Value")

        peaks, properties = find_peaks(smoothed_values)
        peak_heights = smoothed_values[peaks]
        z_scores = (peak_heights - peak_heights.mean()) / peak_heights.std()

        artifact_peaks = peaks[z_scores > 2.0]

        troughs, _ = find_peaks(-smoothed_values)
        
        artifacts = []
        for peak in artifact_peaks:
            left_troughs = troughs[troughs < peak]
            right_troughs = troughs[troughs > peak]

            if len(left_troughs) == 0 or len(right_troughs) == 0:
                continue

            start = left_troughs[-1]
            end = right_troughs[0]

            artifacts.append((start, peak, end))

        for artifact in artifacts:
            start, peak, end = artifact
            print(f"\tStart: {start}")
            print(f"\tPeak: {peak}")
            print(f"\tEnd: {end}")
            print("\t======")

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
    _process_data(patient_id)

if __name__ == "__main__":
    main()