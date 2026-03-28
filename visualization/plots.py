import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RadioButtons
import numpy as np

class VentricleData():
    def __init__(self, 
                 mask, boundary,
                 polar_mask, polar_boundary,
                 center=None):
        self.mask = mask
        self.boundary = boundary

        self.polar_mask = polar_mask
        self.polar_boundary = polar_boundary
        
        self.center = center
        self.transformed_boundary = self._transform_boundary()
    
    def get_data(self, mode: str):
        mode = mode.lower()

        if mode == "cartesian":
            return self.mask, self.boundary
        elif mode == "cartesian_transformed":
            return self.mask, self.transformed_boundary
        elif mode == "polar":
            return self.polar_mask, self.polar_boundary
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
    def _transform_boundary(self):
        rs = self.polar_boundary[:, 0]
        thetas = self.polar_boundary[:, 1]
        cx, cy = self.center[1], self.center[0]

        thetas = np.deg2rad(thetas)

        xs = rs * np.cos(thetas) + cx
        ys = rs * np.sin(thetas) + cy

        transformed_boundary = np.stack((ys, xs), axis=-1)

        return np.array(transformed_boundary)

class ThresholdsData():
    def __init__(self, thresholds,
                 masks, boundaries,
                 polar_masks, polar_boundaries,
                 center=None):
        self.thresholds = thresholds
        thresholds_data: dict[str, VentricleData] = {}
        self.center = center

        for threshold, mask, boundary, polar_mask, polar_boundary in zip(
            thresholds, masks, boundaries, polar_masks, polar_boundaries
        ):
            thresholds_data[threshold] = VentricleData(
                mask=mask,
                boundary=boundary,
                polar_mask=polar_mask,
                polar_boundary=polar_boundary,
                center=self.center
            )
        
        self.thresholds_data = thresholds_data

class ViewData():
    def __init__(self, ct, polar_grad,
                 gt_data: VentricleData, nnunet_data: VentricleData,
                 thresholds_data: ThresholdsData,
                 alg_results: dict,
                 mode: str=None, cmap: str=None):
        self.ct = ct
        self.polar_grad = polar_grad

        self.gt_data = gt_data
        self.nnunet_data = nnunet_data

        self.thresholds_data = thresholds_data

        self.alg_results = alg_results
        center = alg_results["preprocessing"]["center"]
        self.center = np.array(center)

        self.mode = mode
        self.cmap = cmap

    def get_ct_view(self, mode: str):
        mode = mode.lower()

        if mode == "cartesian":
            return self.ct, "grey"
        elif mode == "cartesian_transformed":
            return self.ct, "grey"
        elif mode == "polar":
            return self.polar_grad, "jet"
        else:
            raise ValueError(f"Unknown mode: {mode}")

def _display_normal_data(ax, mask, boundary, color, label):
    ax.scatter(
        boundary[:, 1],
        boundary[:, 0],
        s=5,
        marker='o',
        c=color,
        alpha=0.3,
        label=label
    )

class Viewer:
    def __init__(self, view_data: ViewData):
        self.view_data = view_data

        # checkbutton states
        self.show_gt = True
        self.show_nnunet = True
        self.show_threshold = True

        # radiobutton states
        self.show_all = True
        self.mark_valid = False
        self.show_only_valid = False

        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))

        self._init_widgets()

        self.threshold_info = ""
        self.current_threshold_index = 0
    
    def _init_sliders(self):
        ax_slider = self.fig.add_axes([0.15, 0.10, 0.70, 0.03])

        self.slider = Slider(
            ax=ax_slider,
            label='Threshold Slider',
            valmin=0,
            valmax=len(self.view_data.thresholds_data.thresholds) - 1,
            valinit=0,
            valstep=1
        )

        self.slider.on_changed(self._on_slider_change)
    
    def _init_buttons(self):
        # checkbutton
        ax_check_vis = self.fig.add_axes([0.15, 0.15, 0.30, 0.12])
        ax_check_vis.set_title("Visibility")
        ax_check_vis.axis('off')
        self.check_vis = CheckButtons(
            ax_check_vis,
            labels=["GT", "nnUNet", "Threshold"],
            actives=[True, True, True]
        )

        self.check_vis.on_clicked(self._on_toggle)

        # radiobutton
        ax_radio_val = self.fig.add_axes([0.55, 0.15, 0.30, 0.12])
        ax_radio_val.set_title("Valid Options")
        ax_radio_val.axis('off')
        self.radio_val = RadioButtons(
            ax_radio_val,
            labels=["Show All", "Mark Valid", "Show Only Valid"],
            active=0
        )

        self.radio_val.on_clicked(self._on_radio_click)

    def _init_widgets(self):
        self.fig.subplots_adjust(
            bottom=0.35,
            left=0.1,
            right=0.9,
            top=0.95
        )
        self._init_sliders()
        self._init_buttons()

    def _on_slider_change(self, val):
        self.current_threshold_index = int(val)
        self.render()

    def _on_toggle(self, label):
        if label == "GT":
            self.show_gt = not self.show_gt
        elif label == "nnUNet":
            self.show_nnunet = not self.show_nnunet
        elif label == "Threshold":
            self.show_threshold = not self.show_threshold
        
        self.render()

    def _on_radio_click(self, label):
        self.show_all = (label == "Show All")
        self.mark_valid = (label == "Mark Valid")
        self.show_only_valid = (label == "Show Only Valid")

        self.render()

    def _filter_threshold_boundary(self, threshold, threshold_boundary):
        threshold_data = self.view_data.alg_results["postprocessing"][threshold]
        bp_data = threshold_data["valley_score_data"]["bp_data"]
        
        indices = np.arange(0, 360, 1)
        valid_indices = np.array(bp_data)[:, 0]
        invalid_indices = np.setdiff1d(indices, valid_indices)

        colors = np.full(threshold_boundary.shape[0], 'green')

        if self.show_all:
            return threshold_boundary, colors
        elif self.mark_valid:
            colors[invalid_indices] = 'red'
            return threshold_boundary, colors
        elif self.show_only_valid:
            threshold_boundary = threshold_boundary[valid_indices.astype(int)]
            colors = colors[valid_indices.astype(int)]

            return threshold_boundary, colors
            

    def _display_threshold_data(self, ax, view_data: ViewData, index):
        threshold = view_data.thresholds_data.thresholds[index]
        threshold_data = view_data.thresholds_data.thresholds_data[threshold]
        threshold_mask, threshold_boundary = threshold_data.get_data(mode=view_data.mode)

        if view_data.mode != "cartesian":
            threshold_boundary, colors = self._filter_threshold_boundary(threshold=threshold,
                                                                 threshold_boundary=threshold_boundary)

        ax.scatter(
            threshold_boundary[:, 1],
            threshold_boundary[:, 0],
            s=5,
            marker='o',
            c=colors,
            alpha=0.3,
            label='approximation'
        )

        return threshold

    def render(self):
        fig, ax = self.fig, self.ax

        is_already_rendered = len(ax.images) > 0
        if is_already_rendered:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

        ax.clear()

        mode = self.view_data.mode

        ct_view, cmap = self.view_data.get_ct_view(mode=mode)
        ax.imshow(ct_view, cmap=cmap)

        if mode != "polar":
            center = self.view_data.center
            ax.scatter(
                center[1],
                center[0],
                s=8,
                marker='x',
                c='cyan',
                alpha=0.5
            )

        if self.show_gt:
            gt_mask, gt_boundary = self.view_data.gt_data.get_data(mode=mode)
            _display_normal_data(
                ax=ax,
                mask=gt_mask,
                boundary=gt_boundary,
                color='red',
                label='GT/Doc'
            )
        
        if self.show_nnunet:
            nnunet_mask, nnunet_boundary  = self.view_data.nnunet_data.get_data(mode=mode)
            _display_normal_data(
                ax=ax,
                mask=nnunet_mask,
                boundary=nnunet_boundary,
                color="blue",
                label='nnUNet'
            )
        
        if self.show_threshold:
            self.threshold_info = self._display_threshold_data(
                ax=ax,
                view_data=self.view_data,
                index=self.current_threshold_index
            )

        if is_already_rendered:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        ax.set_title(f"Threshold: {self.threshold_info}")
        ax.legend()
        fig.canvas.draw_idle()

    def show(self):
        self.render()
        plt.show()