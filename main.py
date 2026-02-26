import numpy as np
import matplotlib.pyplot as plt

import os
import yaml
import shutil
import json

import utils
import preprocessor
import thresholds
import postprocessor

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def _save_result_plots(output_dir, threshold):
    preproc_data_dir = os.path.join(output_dir, "preprocessing", "np")
    ct = np.load(os.path.join(preproc_data_dir, "ct.npy"))
    polar_dir_grad = np.load(os.path.join(preproc_data_dir, "polar_dir_grad.npy"))

    doc_mask_b_path = os.path.join(preproc_data_dir, "doc_mask_boundary.npy")
    polar_doc_mask_b_path = os.path.join(preproc_data_dir, "polar_doc_mask_boundary.npy")

    if os.path.exists(doc_mask_b_path):
        doc_mask_b = np.load(doc_mask_b_path)
    else:
        doc_mask_b = None
    
    if os.path.exists(polar_doc_mask_b_path):
        polar_doc_mask_b = np.load(polar_doc_mask_b_path)
    else:
        polar_doc_mask_b = None

    threshold_dir = os.path.join(output_dir, "thresholds", "np", str(threshold))
    approx_mask_b = np.load(os.path.join(threshold_dir, "mask_boundary.npy"))
    approx_polar_mask_b = np.load(os.path.join(threshold_dir, "polar_mask_boundary.npy"))

    #region ORIGINAL RESULT WITHOUT MASK BOUNDARY
    plt.imshow(ct, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "orig.png"))
    plt.close()
    #endregion

    #region ORIGINAL RESULT WITH MASK BOUNDARY
    plt.imshow(ct, cmap='gray')
    if doc_mask_b is not None:
        plt.scatter(
            doc_mask_b[:, 1],
            doc_mask_b[:, 0],
            s=1,
            marker='o',
            c='red',
            alpha=0.3,
            label='Doc/GT'
        )
    plt.scatter(
        approx_mask_b[:, 1],
        approx_mask_b[:, 0],
        s=1,
        marker='o',
        c='green',
        alpha=0.3,
        label='Approximation'
    )
    plt.axis('off')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "orig_result.png"))
    plt.close()
    #endregion

    #region POLAR RESULT WITHOUT MASK BOUNDARY
    plt.imshow(polar_dir_grad, cmap='jet')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "polar.png"))
    plt.close()
    #endregion

    #region POLAR RESULT WITH MASK BOUNDARY
    plt.imshow(polar_dir_grad, cmap='jet')
    if polar_doc_mask_b is not None:
        plt.scatter(
            polar_doc_mask_b[:, 1],
            polar_doc_mask_b[:, 0],
            s=1,
            marker='o',
            c='red',
            alpha=0.3,
            label='Doc/GT'
        )
    plt.scatter(
        approx_polar_mask_b[:, 1],
        approx_polar_mask_b[:, 0],
        s=1,
        marker='o',
        c='green',
        alpha=0.3,
        label='Approximation'
    )
    plt.axis('off')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "polar_result.png"))
    plt.close()
    #endregion


def _save_result_json(output_dir, best_threshold):
    with open(os.path.join(output_dir, "preprocessing", "data.json"), 'r') as f:
        preproc_json = json.load(f)

    with open(os.path.join(output_dir, "thresholds", "score_data.json"), 'r') as f:
        thresholds_json = json.load(f)
    
    ventricle = np.load(os.path.join(output_dir, "preprocessing", "np", "ventricle.npy"))
    ventricle_area = np.sum(ventricle == 1)

    with open(os.path.join(output_dir, "postprocessing", "nms_result.json"), 'r') as f:
        postproc_json = json.load(f)
    
    
    best_mask = np.load(os.path.join(output_dir, "thresholds", "np", str(best_threshold), "mask.npy"))

    
    best_mask_area = np.sum(best_mask == 1)

    area_data = {
        "ventricle": float(ventricle_area),
        "best_mask": float(best_mask_area),
        "ratio": float(best_mask_area / ventricle_area)
    }

    output_data = {
        "best_threshold": float(best_threshold),
        "area_data": area_data
    }

    result_json = {
        "output_data": output_data,
        "preprocessing": preproc_json,
        "thresholds": thresholds_json,
        "postprocessing": postproc_json
    }

    with open(os.path.join(output_dir, "result.json"), 'w') as f:
        json.dump(result_json, f, indent=3)

def _process_one_patient(patient_id: str, patient_data: dict, config: dict):
    output_dir = os.path.join(ROOT_DIR, config["output_dir_name"], patient_id)
    os.makedirs(output_dir, exist_ok=True)

    ct_path = patient_data["ct_path"]
    spacing, ct = utils.scan_to_np_array(ct_path, return_spacing=True)
    
    try:
        doc_mask_path = patient_data["doc_mask_path"]
        doc_mask = utils.scan_to_np_array(doc_mask_path)
        if patient_id == "patient_0001":
            doc_mask = doc_mask[:, :, :, 3]
        elif patient_id == "patient_0010":
            doc_mask = doc_mask[:, :, :, 1]
            doc_mask = (doc_mask == 1).astype(doc_mask.dtype)
        elif patient_id == "patient_0053":
            doc_mask = doc_mask[:, :, :, 2]
            doc_mask = (doc_mask == 1).astype(doc_mask.dtype)
        else:
            doc_mask = doc_mask[:, :, :, 1]
            doc_mask = (doc_mask == 2).astype(doc_mask.dtype)
    except Exception as e:
        print(f"Exception occured during doc mask loading. Setting mask to None.")
        print(f"Exception: {e}")
        doc_mask = None
    
    nnunet_mask_path = patient_data["nnunet_mask_path"]
    nnunet_mask = utils.scan_to_np_array(nnunet_mask_path)
    ventricle = (nnunet_mask == 3).astype(nnunet_mask.dtype)
    atrium = (nnunet_mask == 1).astype(nnunet_mask.dtype)

    lung_mask_path = patient_data.get("lung_mask_path", None)
    if (lung_mask_path is None) or (not os.path.exists(lung_mask_path)):
        patient_data = utils.create_and_save_lung_mask(patient_data=patient_data,
                                        output_dir=output_dir)
    
    lung = utils.scan_to_np_array(patient_data["lung_mask_path"])


    polar_converter = preprocessor.preprocess(
        config=config['preprocessing'],
        ct=ct,
        ct_spacing=spacing,
        atrium=atrium,
        ventricle=ventricle,
        lung=lung,
        output_dir=output_dir,
        doc_mask=doc_mask
    )

    if polar_converter is None:
        return
    
    print("Preprocessing finished successfully!")

    thresholds.create_and_rank_threshold_masks(
        config=config['thresholds'],
        output_dir=output_dir,
        polar_converter=polar_converter
    )
    print("Thresholding finished successfully!")
    
    best_threshold = postprocessor.postprocess(
        config=config["postprocessing"],
        output_dir=output_dir
    )

    print("Postprocessing finished successfully!")

    _save_result_plots(
        output_dir=output_dir,
        threshold=best_threshold
    )

    _save_result_json(
        output_dir=output_dir,
        best_threshold=best_threshold
    )

    if config["cleanup"]:
        for item in os.listdir(output_dir):
            path = os.path.join(output_dir, item)
            if os.path.isdir(path):
                shutil.rmtree(path=path)


def _process_multiple_patients(patients_to_process, patients_data, config):
    for patient_id in patients_to_process:
        print(f"Processing {patient_id}")
        patient_data = patients_data[patient_id]
        _process_one_patient(
            patient_id=patient_id,
            patient_data=patient_data,
            config=config
        )

def main():
    with open(os.path.join(ROOT_DIR, "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)

    with open(os.path.join(ROOT_DIR, "patients_data.json"), 'r') as f:
        patients_data = yaml.safe_load(f)

    patient_id = "patient_0001"
    patient_data = patients_data[patient_id]
    _process_one_patient(
        patient_id=patient_id,
        patient_data=patient_data,
        config=config
    )

    # patients_to_process = []
    # with open(os.path.join(ROOT_DIR, "patients_to_process.txt"), 'r') as file:
    #     for line in file:
    #         line = line.strip()
    #         patients_to_process.append(line)
    # _process_multiple_patients(
    #     patients_to_process=patients_to_process,
    #     patients_data=patients_data,
    #     config=config
    # )


if __name__ == "__main__":
    main()