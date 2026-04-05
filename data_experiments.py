import numpy as np
import utils
from scipy.ndimage import median_filter
from skimage.restoration import denoise_tv_bregman, denoise_tv_chambolle
import matplotlib.pyplot as plt

import os
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _plot_result(title, ct, filtered_ct, cmap):
    ct_std = ct.std()
    filtered_ct_std = filtered_ct.std()

    title = (
        f"{title}\n"
        f"Standard Deviations: {ct_std:.4f} | {filtered_ct_std:.4f}"
    )

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs = axs.flatten()
    for ax in axs:
        ax.axis('off')

    fig.suptitle(title)
    axs[0].imshow(ct, cmap=cmap)
    axs[0].set_title("Original")
    axs[1].imshow(filtered_ct, cmap=cmap)
    axs[1].set_title("Filtered")

    plt.show()

def _plot_median_filter_results(data_dir: str):
    ct = np.load(os.path.join(data_dir, "ct.npy"))
    ventricle_mask = np.load(os.path.join(data_dir, "ventricle.npy"))

    filtered_ct = median_filter(ct, size=(5, 5))

    ct[ventricle_mask == 0] = 0
    filtered_ct[ventricle_mask == 0] = 0

    _plot_result("Median Filter", ct, filtered_ct, "gray")

def _plot_tv_results(data_dir: str):
    ct = np.load(os.path.join(data_dir, "ct.npy"))
    ventricle_mask = np.load(os.path.join(data_dir, "ventricle.npy"))

    ct_min = ct.min()
    ct_max = ct.max()
    ct = (ct - ct_min) / (ct_max - ct_min)

    filtered_ct = denoise_tv_bregman(ct, weight=10.0)

    ct = ct * (ct_max - ct_min) + ct_min
    filtered_ct = filtered_ct * (ct_max - ct_min) + ct_min

    ct[ventricle_mask == 0] = 0
    filtered_ct[ventricle_mask == 0] = 0

    _plot_result("TV Bregman", ct, filtered_ct, "gray")

def main():
    patient_id = "patient_0002"
    data_dir = os.path.join("vis_output", patient_id, "preprocessing", "np")

    # _plot_median_filter_results(data_dir=data_dir)
    _plot_tv_results(data_dir=data_dir)

if __name__ == "__main__":
    main()