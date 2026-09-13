import numpy as np
import pandas as pd

import os
import sys
import time
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

def dice(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()

    if total == 0:
        return 1.0
    
    return 2.0 * intersection / total

def main():
    dir = r"gac_exp_output_ep_1\patient_0001"

    mask_name = "c1p00_p1p00.seg.nrrd"

    mask_datas = []

    for it_number in os.listdir(dir):
        mask = utils.scan_to_np_array(os.path.join(dir, it_number, mask_name))

        mask_datas.append(
            {
                "it": it_number,
                "mask": mask
            }
        )

    n = len(mask_datas)
    labels = [m["it"] for m in mask_datas]
    dice_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            d = dice(mask_datas[i]["mask"], mask_datas[j]["mask"])
            dice_matrix[i, j] = d
            dice_matrix[j, i] = d

    dice_df = pd.DataFrame(dice_matrix, index=labels, columns=labels)

    print(dice_df)

if __name__ == "__main__":
    main()