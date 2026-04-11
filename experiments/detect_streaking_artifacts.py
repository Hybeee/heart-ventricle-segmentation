import numpy as np
import matplotlib.pyplot as plt
import os

def _plot_1d_func(values, xlabel, ylabel):
    plt.plot(values)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def main():
    patient_dir = os.path.join("vis_output", "patient_0045")

    grad_map_path = os.path.join(patient_dir, "preprocessing", "np", "polar_dir_grad.npy")
    grad = np.load(grad_map_path)
    
    ventricle_boundary_path = os.path.join(patient_dir, "preprocessing", "np", "polar_ventricle_boundary.npy")
    ventricle_boundary = np.load(ventricle_boundary_path)

    radii = ventricle_boundary[:, 0].astype(int)
    theta = ventricle_boundary[:, 1].astype(int)

    values = grad[radii, theta]

    _plot_1d_func(values, "Radius", "Value")

if __name__ == "__main__":
    main()