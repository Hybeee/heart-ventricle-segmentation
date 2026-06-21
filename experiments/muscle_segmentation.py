import numpy as np
from skimage.morphology import convex_hull_image
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import SimpleITK as sitk

import os

import utils

def main():
    patient_id = "patient_0008"
    output_path = os.path.join("streaking_viewer_output", patient_id)

    ct = utils.scan_to_np_array(scan_path=os.path.join(output_path, "ct.nii.gz"))
    mask_sitk, mask = utils.scan_to_np_array(scan_path=os.path.join(output_path, "final_mask_nip.seg.nrrd"), return_sitk=True)

    ct = ct[238, :, :]
    mask = mask[238, :, :]

    chull = convex_hull_image(mask)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ax = axes.ravel()

    ax[0].set_title('Original mask picture')
    ax[0].imshow(mask, cmap=plt.cm.gray)
    ax[0].set_axis_off()

    ax[1].set_title('Convex hull picture')
    ax[1].imshow(chull, cmap=plt.cm.gray)
    ax[1].set_axis_off()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()