import matplotlib.pyplot as plt

class VentricleData():
    def __init__(self, 
                 mask, boundary,
                 polar_mask, polar_boundary):
        self.mask = mask
        self.boundary = boundary

        self.polar_mask = polar_mask
        self.polar_boundary = polar_boundary

class ThresholdsData():
    def __init__(self, thresholds,
                 masks, boundaries,
                 polar_masks, polar_boundaries):
        thresholds_data = {}

        for threshold, mask, boundary, polar_mask, polar_boundary in zip(
            thresholds, masks, boundaries, polar_masks, polar_boundaries
        ):
            thresholds_data[threshold] = VentricleData(
                mask=mask,
                boundary=boundary,
                polar_mask=polar_mask,
                polar_boundary=polar_boundary
            )
        
        self.thresholds_data = thresholds_data

class ViewData():
    def __init__(self, ct, polar_grad,
                 gt_data: VentricleData, nnunet_data: VentricleData,
                 thresholds_data: ThresholdsData,
                 alg_results: dict):
        self.ct = ct
        self.polar_grad = polar_grad

        self.gt_data = gt_data
        self.nnunet_data = nnunet_data

        self.thresholds_data = thresholds_data

        self.alg_results = alg_results

def view_ct():
    pass