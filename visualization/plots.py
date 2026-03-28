import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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
        center = np.array(center)
        print(f"Center: {center}")

        self.gt_data.center = center
        self.nnunet_data.center = center
        self.thresholds_data.center = center

        self.mode = mode
        self.cmap = cmap

    def get_ct_view(self, mode: str):
        mode = mode.lower()

        if mode == "cartesian":
            return self.ct
        elif mode == "cartesian_transformed":
            return self.ct
        elif mode == "polar":
            return self.polar_grad
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

def _display_threshold_data(ax, view_data: ViewData, index):
    threshold = view_data.thresholds_data.thresholds[index]
    threshold_data = view_data.thresholds_data.thresholds_data[threshold]
    threshold_mask, threshold_boundary = threshold_data.get_data(mode=view_data.mode)

    ax.scatter(
        threshold_boundary[:, 1],
        threshold_boundary[:, 0],
        s=5,
        marker='o',
        c='green',
        alpha=0.3,
        label='approximation'
    )

    return threshold

def display_images_on_ax(fig, ax, view_data: ViewData, index):
    mode = view_data.mode
    gt_mask, gt_boundary = view_data.gt_data.get_data(mode=mode)
    nnunet_mask, nnunet_boundary  = view_data.nnunet_data.get_data(mode=mode)
    
    ax.clear()

    ct_view = view_data.get_ct_view(mode=mode)
    ax.imshow(ct_view, cmap=view_data.cmap)


    _display_normal_data(
        ax=ax,
        mask=gt_mask,
        boundary=gt_boundary,
        color="red",
        label='GT/Doc'
    )

    _display_normal_data(
        ax=ax,
        mask=nnunet_mask,
        boundary=nnunet_boundary,
        color="blue",
        label='nnUNet'
    )

    threshold_info = _display_threshold_data(
        ax=ax,
        view_data=view_data,
        index=index
    )

    ax.set_title(f"Threshold: {threshold_info}")
    ax.legend()
    fig.canvas.draw()

    plt.show()

# NOTE: will be used for view_ct and view_polar
#       probably will use a different func for view_grad_map()
def view_threshold_masks(view_data: ViewData):
    initial_threshold_index = 0

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    axthreshold = fig.add_axes([0.2, 0.01, 0.65, 0.03])
    threshold_slider = Slider(
        ax=axthreshold,
        label='Threshold Slider',
        valmin=0,
        valmax=len(view_data.thresholds_data.thresholds) - 1,
        valinit=initial_threshold_index,
        valstep=1
    )

    threshold_slider.on_changed(
        lambda index: display_images_on_ax(fig, ax, view_data, int(index))
    )

    display_images_on_ax(
        fig,
        ax,
        view_data,
        int(initial_threshold_index)
    )

def view_ct(view_data: ViewData):
    view_threshold_masks(view_data=view_data)

def view_polar():
    pass

def view_1d_grad_map():
    pass