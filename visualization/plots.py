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

class Viewer2D:
    def __init__(self, view_data: ViewData):
        self.view_data = view_data

        # checkbutton states
        self.show_gt = True
        self.show_nnunet = True
        self.show_threshold = True
        self.show_at_theta_only = False
        self.theta_draw_line = False

        # radiobutton states
        self.show_all = True
        self.mark_valid = False
        self.show_only_valid = False

        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))

        self._init_widgets()

        self.threshold_info = ""
        self.current_threshold_index = 0
        self.current_theta_index = 0
    
    def _init_sliders(self):
        ax_threshold_slider = self.fig.add_axes([0.20, 0.10, 0.70, 0.03])

        self.threshold_slider = Slider(
            ax=ax_threshold_slider,
            label='Threshold Slider',
            valmin=0,
            valmax=len(self.view_data.thresholds_data.thresholds) - 1,
            valinit=0,
            valstep=1
        )

        self.threshold_slider.on_changed(
            lambda val: self._on_slider_change(threshold_index=val, current_theta_index=self.current_theta_index)
        )

        ax_theta_slider = self.fig.add_axes([0.20, 0.05, 0.70, 0.03])

        thresholds_data = self.view_data.thresholds_data.thresholds_data
        threshold_data = list(thresholds_data.values())[0]
        valmax = threshold_data.polar_boundary.shape[0]

        self.theta_slider = Slider(
            ax=ax_theta_slider,
            label='Theta Slider (Index)',
            valmin=0,
            valmax=valmax - 1,
            valinit=0,
            valstep=1
        )

        self.theta_slider.on_changed(
            lambda val: self._on_slider_change(threshold_index=self.current_threshold_index, current_theta_index=val)
        )
    
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
        ax_radio_val = self.fig.add_axes([0.35, 0.15, 0.30, 0.12])
        ax_radio_val.set_title("Valid Options")
        ax_radio_val.axis('off')
        self.radio_val = RadioButtons(
            ax_radio_val,
            labels=["Show All", "Mark Valid", "Show Only Valid"],
            active=0
        )

        self.radio_val.on_clicked(self._on_radio_click)

        # checkbutton
        ax_check_theta = self.fig.add_axes([0.55, 0.15, 0.30, 0.12])
        ax_check_theta.set_title("Show At Theta Only")
        ax_check_theta.axis('off')
        self.check_theta = CheckButtons(
            ax_check_theta,
            labels=["Show Boundary Point At Theta Only", "Draw Line"],
            actives=[False, False]
        )

        self.check_theta.on_clicked(self._on_toggle)

    def _init_widgets(self):
        self.fig.subplots_adjust(
            bottom=0.35,
            left=0.1,
            right=0.9,
            top=0.95
        )
        self._init_sliders()
        self._init_buttons()

    def _on_slider_change(self, threshold_index, current_theta_index):
        self.current_threshold_index = threshold_index
        self.current_theta_index = current_theta_index

        self.render()

    def _on_toggle(self, label):
        if label == "GT":
            self.show_gt = not self.show_gt
        elif label == "nnUNet":
            self.show_nnunet = not self.show_nnunet
        elif label == "Threshold":
            self.show_threshold = not self.show_threshold
        elif label == "Show Boundary Point At Theta Only":
            self.show_at_theta_only = not self.show_at_theta_only
        elif label == "Draw Line":
            self.theta_draw_line = not self.theta_draw_line
        
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

        if self.show_at_theta_only:
            point = threshold_boundary[self.current_theta_index]
            color = 'red' if self.current_theta_index in invalid_indices else 'green'

            return np.array([point]), np.array([color])

        elif self.show_all:
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
            label='Approximation'
        )

        if self.show_at_theta_only and self.theta_draw_line:
            points = np.array([threshold_boundary[0], self.view_data.center])
            xs = points[:, 1]
            ys = points[:, 0]
            
            ax.plot(
                xs,
                ys,
                color='green',
                linewidth=1,
                alpha=0.5
            )

        return threshold
    
    def _display_normal_data(self, ax, mask, boundary, color, label):
        ax.scatter(
            boundary[:, 1],
            boundary[:, 0],
            s=5,
            marker='o',
            c=color,
            alpha=0.3,
            label=label
        )

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
            self._display_normal_data(
                ax=ax,
                mask=gt_mask,
                boundary=gt_boundary,
                color='red',
                label='GT/Doc'
            )
        
        if self.show_nnunet:
            nnunet_mask, nnunet_boundary  = self.view_data.nnunet_data.get_data(mode=mode)
            self._display_normal_data(
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

class Viewer1D:
    def __init__(self, view_data: ViewData):
        self.view_data = view_data
        self.valley_data = ValleyData(self.view_data.polar_grad)

        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))

        self._init_widgets()

        # checkbutton states
        self.show_valleys = True

        # radiobutton states
        self.valley_points = True
        self.valley_lines = False

        self.title = ""
        self.current_threshold_index = 0
        self.current_theta_index = 0

    def _init_sliders(self):
        ax_threshold_slider = self.fig.add_axes([0.20, 0.10, 0.70, 0.03])

        self.threshold_slider = Slider(
            ax=ax_threshold_slider,
            label='Threshold Slider',
            valmin=0,
            valmax=len(self.view_data.thresholds_data.thresholds) - 1,
            valinit=0,
            valstep=1
        )

        self.threshold_slider.on_changed(
            lambda val: self._on_slider_change(threshold_index=val, current_theta_index=self.current_theta_index)
        )

        ax_theta_slider = self.fig.add_axes([0.20, 0.05, 0.70, 0.03])

        thresholds_data = self.view_data.thresholds_data.thresholds_data
        threshold_data = list(thresholds_data.values())[0]
        valmax = threshold_data.polar_boundary.shape[0]

        self.theta_slider = Slider(
            ax=ax_theta_slider,
            label='Theta Slider (Index)',
            valmin=0,
            valmax=valmax - 1,
            valinit=0,
            valstep=1
        )

        self.theta_slider.on_changed(
            lambda val: self._on_slider_change(threshold_index=self.current_threshold_index, current_theta_index=val)
        )

    def _init_buttons(self):
        ax_check_valley_display = self.fig.add_axes([0.15, 0.20, 0.30, 0.08])
        ax_check_valley_display.set_title("Valley Info")
        ax_check_valley_display.axis('off')

        # checkbutton
        self.check_valley_display = CheckButtons(
            ax_check_valley_display,
            labels=["Show Valleys"],
            actives=[True]
        )
        self.check_valley_display.on_clicked(self._on_toggle)

        # radiobutton
        ax_radio_valley_display = self.fig.add_axes([0.15, 0.15, 0.30, 0.08])
        ax_radio_valley_display.axis('off')

        self.radio_valley_display = RadioButtons(
            ax_radio_valley_display,
            labels=["Display as points", "Display as lines"],
            active=0
        )
        self.radio_valley_display.on_clicked(self._on_radio_click)

    def _init_widgets(self):
        self.fig.subplots_adjust(
            bottom=0.35,
            left=0.1,
            right=0.9,
            top=0.95
        )

        self._init_sliders()
        self._init_buttons()

    def _on_slider_change(self, threshold_index, current_theta_index):
        self.current_threshold_index = threshold_index
        self.current_theta_index = current_theta_index

        self.render()

    def _on_toggle(self, label):
        if label == "Show Valleys":
            self.show_valleys = not self.show_valleys
        
        self.render()

    def _on_radio_click(self, label):
        self.valley_points = (label == "Display as points")
        self.valley_lines = (label == "Display as lines")

        self.render()

    def _display_normal_data(self, ax, grad_map_1d, point, color, label):
        radius = point[0]
        value = grad_map_1d[radius]

        ax.scatter(
            radius,
            value,
            s=20,
            c=color,
            marker='o',
            label=label
        )

    def _get_threshold_point(self):
        threshold = self.view_data.thresholds_data.thresholds[self.current_threshold_index]
        self.title = threshold
        
        threshold_data = self.view_data.thresholds_data.thresholds_data[threshold]
        point = threshold_data.polar_boundary[self.current_theta_index]

        return point

    def _display_valleys(self, ax, grad_map_1d, max_r):
        valley_locations = self.valley_data.valley_positions[self.current_theta_index]
        valley_locations = np.array(valley_locations)
        valley_rs = valley_locations[valley_locations < max_r]

        valley_values = grad_map_1d[valley_rs]

        if self.valley_points:
            ax.scatter(
                valley_rs,
                valley_values,
                s=10,
                c='purple',
                marker='o',
            )
        elif self.valley_lines:
            for valley_loc in valley_rs:
                r_pair = [valley_loc, valley_loc + 1]
                value_pair = grad_map_1d[valley_loc:valley_loc+2]

                ax.plot(
                    r_pair,
                    value_pair,
                    color='purple',
                    alpha=0.5,
                    linewidth=2,
                )

                ax.scatter(
                    r_pair,
                    value_pair,
                    s=10,
                    c='purple',
                    marker='o',
                )

    def render(self):
        fig, ax = self.fig, self.ax

        ax.clear()

        grad_map_1d = self.view_data.polar_grad[:, self.current_theta_index]
    
        gt_point = self.view_data.gt_data.polar_boundary[self.current_theta_index]
        nnunet_point = self.view_data.nnunet_data.polar_boundary[self.current_theta_index]
        threshold_point = self._get_threshold_point()

        max_bp = max(gt_point[0], nnunet_point[0], threshold_point[0])
        max_r = min(max_bp + 15, len(grad_map_1d)-1)
        grad_map_1d = grad_map_1d[:max_r]

        ax.plot(grad_map_1d)

        self._display_normal_data(ax, grad_map_1d, gt_point, 'red', 'GT/Doc')
        self._display_normal_data(ax, grad_map_1d, nnunet_point, 'blue', 'nnUNet')
        self._display_normal_data(ax, grad_map_1d, threshold_point, 'green', 'Approximation')
        if self.show_valleys:
            self._display_valleys(ax, grad_map_1d, max_r)

        ax.set_xlabel("Radii")
        ax.set_ylabel("Gradient Value")
        ax.set_title(self.title)
        ax.legend()

        fig.canvas.draw_idle()

    def show(self):
        self.render()
        plt.show()