import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

import utils

import os
import json
import multiprocessing as mp

class Viewer:
    def __init__(self, ct, mask, nnunet_mask, sigma, mode):
        self.ct = ct
        self.mask = mask
        self.nnunet_mask = nnunet_mask
        self.sigma = sigma
        self.mode=mode

        # initializing
        self._init()

    def _get_bbox_middle_slice_z(self, mask):
        z_dim = mask.shape[0]

        for z in range(z_dim):
            if np.any(mask[z, :, :] == 1):
                z_start = z
                break

        for z in reversed(range(z_dim)):
            if np.any(mask[z, :, :] == 1):
                z_end = z
                break

        z_middle = (z_start + z_end) // 2

        return (z_start, z_middle, z_end)

    def _create_polar_converter(self, center, ct):
        """
        Expects 2D slices.
        """

        maxRadius = min(
            center[0],
            center[1],
            ct.shape[0] - center[0] - 50,
            ct.shape[1] - center[1] - 50
        )

        radius_samples = maxRadius
        angle_samples = 360
        polar_converter = utils.PolarConverter(
            dsize=(radius_samples, angle_samples),
            maxRadius=maxRadius,
            center=(center[1], center[0]),
            flags=cv2.WARP_POLAR_LINEAR
        )

        return polar_converter

    def _set_slice_data(self):
        self.curr_ct_slice = self.ct[self.slice_index, :, :]
        self.curr_mask_slice = self.mask[self.slice_index, :, :]
        self.curr_nnunet_mask_slice = self.nnunet_mask[self.slice_index, :, :]

        if self.mode == "polar":
            self.slice_center = utils.calculate_slice_center(slice=self.curr_nnunet_mask_slice)

            self.slice_pc = self._create_polar_converter(
                center=self.slice_center,
                ct=self.curr_ct_slice
            )

            _, self.polar_dir_grad_slice = utils.get_directional_grad(
                ct=self.curr_ct_slice,
                center=self.slice_center,
                sigma=self.sigma,
                polarConverter=self.slice_pc
            )
            self.polar_dir_grad_slice = utils.clip_polar_gradients(polar_dir_grad_ct=self.polar_dir_grad_slice)

            self.polar_mask_slice = self.slice_pc.cv2WarpPolar(image=self.curr_mask_slice.astype(np.int32))
            self.polar_mask_slice = (self.polar_mask_slice != 0).astype(self.polar_mask_slice.dtype)
            self.polar_mask_slice_b = utils.get_polar_boundary_points(polar_mask=self.polar_mask_slice, theta_step_size=1)

    def _on_slider_changed(self, slice_index):
        self.slice_index = slice_index
        self._set_slice_data()

        self.render()

    def _init_sliders(self):
        ax_slice_index_slider = self.fig.add_axes([0.20, 0.10, 0.70, 0.03])

        self.slice_index_slider = Slider(
            ax=ax_slice_index_slider,
            label='Slice Index Slider',
            valmin=0,
            valmax=self.mask.shape[0] - 1,
            valinit=self.slice_index,
            valsetp=1
        )

        self.slice_index_slider.on_chagned(
            lambda val: self._on_slider_changed(slice_index=val)
        )

    def _init_widgets(self):
        self._init_sliders()

    def _init(self):
        self.start, self.middle, self.end = self._get_bbox_middle_slice_z(self.mask)
        self.slice_index = self.middle

        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        self._init_widgets()

        self._set_slice_data()
        self.render()

    def render(self):
        fig, ax = self.fig, self.ax

        is_already_rendered = len(ax.images) > 0
        if is_already_rendered:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

        if self.mode == "ct":
            ct_display = self.curr_ct_slice
            mask_display = self.curr_mask_slice

            ax.imshow(ct_display, cmap='gray')
            ax.scatter(
                self.slice_center[1],
                self.slice_center[0],
                s=8,
                marker='x',
                c='cyan',
                alpha=0.7
            )
            ax.imshow(mask_display, cmap='Blues', alpha=0.5)
        else:
            ct_display = self.polar_dir_grad_slice
            mask_display_b = self.polar_mask_slice_b

            ax.imshow(ct_display, cmap='jet')
            ax.scatter(
                mask_display_b[:, 1],
                mask_display_b[:, 0],
                s=5,
                marker='o',
                c='green',
                alpha=0.3
            )

        if is_already_rendered:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        
        fig.canvas.draw_idle()

    def view(self):
        self.render()
        plt.show()

def main():
    output_path = ""

    with open(os.path.join(output_path, "results.json"), 'r') as f:
        results = json.load(f)
    sigma = results["sigma"]

    ct = utils.scan_to_np_array(scan_path=os.path.join(output_path, "ct.nii.gz"))
    mask = utils.scan_to_np_array(scan_path=os.path.join(output_path, "final_mask.seg.nrrd"))
    nnunet_mask = utils.scan_to_np_array(scan_path=os.path.join(output_path, "nnunet_mask.seg.nrrd"))

    ct_viewer = Viewer(
        ct=ct,
        mask=mask,
        nnunet_mask=nnunet_mask,
        sigma=sigma,
        mode="ct"
    )

    polar_viewer = Viewer(
        ct=ct,
        mask=mask,
        nnunet_mask=nnunet_mask,
        sigma=sigma,
        mode="polar"
    )

    p1 = mp.Process(target=ct_viewer.view)
    p2 = mp.Process(target=polar_viewer.view)

    p1.start()
    p2.start()

    p1.join()
    p2.join()


if __name__ == "__main__":
    main()