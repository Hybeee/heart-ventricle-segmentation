import SimpleITK as sitk
import numpy as np

import os
import sys
import time
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils

def _get_edge_potential(mask_sitk, sigma=2.0):
    signed_distance = sitk.SignedMaurerDistanceMap(
        mask_sitk,
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True
    )

    signed_distance_np = sitk.GetArrayFromImage(signed_distance)
    edge_potential_np = 1.0 - np.exp(-np.abs(signed_distance_np) / sigma)

    return edge_potential_np

def _process_patient(data_dir, patient_id, output_dir):
    patient_dir = os.path.join(data_dir, patient_id)
    patient_output_dir = os.path.join(output_dir, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)

    files_to_copy = ["ct.nii.gz", "final_mask_nip.seg.nrrd"]

    for file in files_to_copy:
        shutil.copyfile(
            src=os.path.join(patient_dir, file),
            dst=os.path.join(patient_output_dir, file)
        )

    mask_sitk, mask = utils.scan_to_np_array(scan_path=os.path.join(patient_dir, "final_mask_nip.seg.nrrd"), return_sitk=True)
    
    edge_potential_np = _get_edge_potential(mask_sitk=mask_sitk)

    utils.save_data(
        data=edge_potential_np,
        ref_sitk=mask_sitk,
        output_dir=patient_output_dir,
        name="edge_potential",
        is_mask=False
    )

def main():
    data_dir = f"{ROOT_DIR}/pipeline_output"
    patient_id = "patient_0001"

    output_dir = f"{ROOT_DIR}/edge_potential_output"
    os.makedirs(output_dir, exist_ok=True)

    _process_patient(
        data_dir=data_dir,
        patient_id=patient_id,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()