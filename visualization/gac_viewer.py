import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.measure import perimeter
import matplotlib.colors as mcolors
import numpy as np

import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

def get_bbox_interval_y(mask):
    y_dim = mask.shape[1]

    for y in range(y_dim):
        if np.any(mask[:, y, :] == 1):
            y_start = y
            break

    for y in reversed(range(y_dim)):
        if np.any(mask[:, y, :] == 1):
            y_end = y
            break

    y_middle = (y_start + y_end) // 2

    return (y_start, y_middle, y_end)

class GACViewer:
    def __init__(self,
                 ct, mask,
                 data_dir, mask_param_sets,
                 iterations):
        self.ct = ct
        self.mask = mask
        self.data_dir = data_dir
        self.mask_param_sets = mask_param_sets
        self.iterations = iterations

        # slider
        self.start, self.slice_index, self.end = get_bbox_interval_y(self.mask)
        self.iteration_index = 0
        self.alpha = 0.7
        self.param_set_index = 0

        self.mask_names = [mask_param_set["name"] for mask_param_set in self.mask_param_sets]
        for i, mask_name in enumerate(self.mask_names):
            print(f"{mask_name} - {i}")

        self.gac_mask_paths = self._load_gac_mask_paths()

        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        self._init_widgets()

    def _load_gac_mask_paths(self):
        mask_paths = []

        mask_name = self.mask_names[self.param_set_index]
        for iteration in self.iterations:
            mask_path = os.path.join(self.data_dir, f"it{iteration}", f"{mask_name}.seg.nrrd")
            # mask = utils.scan_to_np_array(mask_path)
            mask_paths.append(mask_path)

        return mask_paths

    def _on_param_set_change(self, val):
        self.param_set_index = int(val)
        self.gac_mask_paths = self._load_gac_mask_paths()
        self.iteration_index = 0
        self.iteration_slider.set_val(0)

    def _on_slice_change(self, val):
        self.slice_index = int(val)
        self._render()

    def _on_iteration_change(self, val):
        self.iteration_index = int(val)
        self._render()

    def _on_alpha_change(self, val):
        self.alpha = val
        self._render()

    def _init_sliders(self):
        param_ax = self.fig.add_axes([0.3, 0.22, 0.5, 0.03])
        self.param_slider = Slider(
            ax=param_ax,
            label="Param set",
            valmin=0,
            valmax=len(self.mask_names) - 1,
            valinit=self.param_set_index,
            valstep=1
        )
        self.param_slider.on_changed(self._on_param_set_change)

        slice_ax = self.fig.add_axes([0.3, 0.16, 0.5, 0.03])
        self.slice_slider = Slider(
            ax=slice_ax,
            label="Slice",
            valmin=self.start,
            valmax=self.end,
            valinit=self.slice_index,
            valstep=1,
        )
        self.slice_slider.on_changed(self._on_slice_change)

        iteration_ax = self.fig.add_axes([0.3, 0.10, 0.5, 0.03])
        self.iteration_slider = Slider(
            ax=iteration_ax,
            label="Iteration",
            valmin=0,
            valmax=len(self.iterations) - 1,
            valinit=self.iteration_index,
            valstep=1,
        )
        self.iteration_slider.on_changed(self._on_iteration_change)

        alpha_ax = self.fig.add_axes([0.3, 0.04, 0.5, 0.03])
        self.alpha_slider = Slider(
            ax=alpha_ax,
            label="Alpha",
            valmin=0.0,
            valmax=1.0,
            valinit=self.alpha,
        )
        self.alpha_slider.on_changed(self._on_alpha_change)

    def _init_widgets(self):
        self.fig.subplots_adjust(bottom=0.3)

        self._init_sliders()

    def _render(self):
        ct_slice = np.flipud(self.ct[:, self.slice_index, :])
        mask_slice = np.flipud(self.mask[:, self.slice_index, :])
        mask_b = utils.get_mask_boundary(mask_slice)
        gac_mask = utils.scan_to_np_array(self.gac_mask_paths[self.iteration_index])
        gac_mask_slice = np.flipud(gac_mask[:, self.slice_index, :])

        fig, ax = self.fig, self.ax

        is_already_rendered = len(ax.images) > 0
        if is_already_rendered:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

        ax.clear()

        curr_set = self.mask_param_sets[self.param_set_index]
        ratio = curr_set["curvature scaling"] / curr_set["advection scaling"] if curr_set["propagation scaling"] != 0 else np.nan
        title = (
            f"iteration={self.iterations[self.iteration_index]}"
            + f"\nname={curr_set['name']}"
            + f"\ncurv={curr_set['curvature scaling']}   prop={curr_set['propagation scaling']}   adv={curr_set.get('advection scaling', np.nan)}"
            + f"\nratio={ratio:.4f}"
        )
        ax.set_title(title)

        ax.imshow(ct_slice, cmap='gray')
        if mask_b.size > 0:
            ax.scatter(
                mask_b[:, 1],
                mask_b[:, 0],
                s=3,
                c='green',
                marker='o',
                alpha=0.7
            )
        ax.imshow(gac_mask_slice, cmap='Blues', alpha=self.alpha)

        if is_already_rendered:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        fig.canvas.draw_idle()

    def view(self):
        self._render()
        plt.show()

def _get_mask_boundary_len(mask):
    import scipy.ndimage as ndimage
    mask_e = ndimage.binary_erosion(mask, structure=np.ones((3, 3)))
    boundary = mask ^ mask_e
    points = np.argwhere(boundary)

    return len(points)

def _load_iteration_data(gac_dir, it_dir_names, name, slice_index, value_type):
    data = []

    for it_dir in it_dir_names:
        mask_path = os.path.join(gac_dir, it_dir, f"{name}.seg.nrrd")
        mask = utils.scan_to_np_array(mask_path)
        mask = mask[:, slice_index, :]

        if value_type.lower() == "area":
            data.append(np.sum(mask))
        elif value_type.lower() == "length":
            data.append(_get_mask_boundary_len(mask))
        else:
            raise ValueError(f"Unknown value type: value type={value_type}")

    return data

def view_iteration_data(gac_dir, mask_param_sets, slice_index, init_value=None, value_type="area"):
    it_dir_names = os.listdir(gac_dir)
    it_dir_names = sorted(it_dir_names, key=lambda d: int(d[2:]))
    iterations = [int(d[2:]) for d in it_dir_names]

    param_data = {}

    for mask_param_set in mask_param_sets:
        name = mask_param_set["name"]
        curv = mask_param_set["curvature scaling"]
        prop = mask_param_set["propagation scaling"]

        values = _load_iteration_data(gac_dir, it_dir_names, name, slice_index, value_type)

        param_data[name] = {
            "values": values,
            "ratio": curv / prop if prop != 0 else np.nan
        }

    fig, ax = plt.subplots(figsize=(9, 6))

    finite_items = {n: d for n, d in param_data.items() if np.isfinite(d["ratio"])}
    infinite_items = {n: d for n, d in param_data.items() if not np.isfinite(d["ratio"])}
    ratios = [d["ratio"] for d in finite_items.values()]
    norm = mcolors.LogNorm(vmin=min(ratios), vmax=max(ratios))
    cmap = plt.cm.coolwarm

    for name, d in sorted(finite_items.items(), key=lambda kv: kv[1]["ratio"]):
        color = cmap(norm(d["ratio"]))
        ax.plot(iterations, d["values"], color=color, linewidth=1.5, alpha=0.85)

    for name, d in infinite_items.items():
        ax.plot(iterations, d["values"], color="black", linestyle="--", linewidth=2,
                label=f"{name} (prop=0, ratio=infinite)")

    if init_value is not None:
        ax.axhline(init_value, color="black", linestyle="--", linewidth=1, label="initial value")
        ax.legend()

    ax.set_yscale("linear")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(value_type)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="ratio (curv/prop)")

    fig.tight_layout()
    plt.show()

def build_mask_param_sets(gac_dir):
    param_names = {
        "c": "curvature scaling",
        "p": "propagation scaling",
        "a": "advection scaling"
    }
    mask_param_sets = []

    it_dir = os.listdir(gac_dir)[0]

    for mask_name in os.listdir(os.path.join(gac_dir, it_dir)):
        mask_param_set = {}

        mask_name = mask_name[:-9]
        mask_param_set["name"] = mask_name

        param_datas = mask_name.split("_")
        for data in param_datas:
            param_type = data[0]
            param_name = param_names[param_type]

            param_value = data[1:]
            param_value = float(param_value.replace("p", "."))

            mask_param_set[param_name] = param_value

        mask_param_sets.append(mask_param_set)

    return mask_param_sets


def main():
    data_dir = os.path.join(ROOT_DIR, "pipeline_output")
    gac_dir = os.path.join(ROOT_DIR, "heart_muscle_segmentation_convergence_output_p8")

    patient_id = "patient_0001"
    patient_data_dir = os.path.join(data_dir, patient_id)
    patient_gac_dir = os.path.join(gac_dir, patient_id)

    mask_param_sets = build_mask_param_sets(gac_dir=patient_gac_dir)
    iterations = sorted(int(it_name[2:]) for it_name in os.listdir(patient_gac_dir) if os.path.isdir(os.path.join(patient_gac_dir, it_name)))

    ct = utils.scan_to_np_array(os.path.join(patient_data_dir, "ct.nii.gz"))
    mask = utils.scan_to_np_array(os.path.join(patient_data_dir, "final_mask_nip.seg.nrrd"))

    slice_index = 201
    value_type = "area"
    if value_type.lower() == "area":
        init_value = np.sum(mask[:, slice_index, :])
    elif value_type.lower() == "length":
        init_value = None
    else:
        raise ValueError(f"Unknown value type: value type={value_type}")

    # view_iteration_data(
    #     gac_dir=patient_gac_dir,
    #     mask_param_sets=mask_param_sets,
    #     slice_index=slice_index,
    #     init_value=init_value,
    #     value_type=value_type
    # )

    viewer = GACViewer(
        ct=ct,
        mask=mask,
        data_dir=patient_gac_dir,
        mask_param_sets=mask_param_sets,
        iterations=iterations
    )

    viewer.view()

if __name__ == "__main__":
    main()