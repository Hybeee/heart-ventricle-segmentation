import SimpleITK as sitk

import os, sys
ROOT_DIR = os.path.dirname((os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def main():
    data_dir = os.path.join(ROOT_DIR, "postproc_alg_vars_output")
    nnunet_inf_dir = os.path.join(ROOT_DIR, "nnunet_finetune/nnUNet_results/Dataset002_LV_Doc/nnUNetTrainer_250epochs__nnUNetPlans_ft__3d_fullres")

    folds = [0, 1, 2, 3, 4]

    output_dir = os.path.join(ROOT_DIR, "nnunet_inference_output_collection")

    for fold in folds:
        val_dir = os.path.join(nnunet_inf_dir, f"fold_{fold}", "validation")
        for patient_id in os.listdir(val_dir):
            if not patient_id.endswith(".nii.gz"):
                continue

            patient_id = patient_id[:-len(".nii.gz")]
            patient_output_dir = os.path.join(output_dir, patient_id)
            os.makedirs(patient_output_dir, exist_ok=True)

            ct = sitk.ReadImage(os.path.join(data_dir, patient_id, "ct.nii.gz"))
            doc_mask = sitk.ReadImage(os.path.join(data_dir, patient_id, "doc_mask.seg.nrrd"))
            th_alg_mask = sitk.ReadImage(os.path.join(data_dir, patient_id, "final_mask_nip.seg.nrrd"))
            nnunet_inf_mask = sitk.ReadImage(os.path.join(val_dir, f"{patient_id}.nii.gz"))
            nnunet_inf_all_mask = sitk.ReadImage(os.path.join(nnunet_inf_dir, "fold_all", "validation", f"{patient_id}.nii.gz"))

            datas = [
                [ct, "ct.nii.gz"],
                [doc_mask, "doc_mask.seg.nrrd"],
                [th_alg_mask, "final_mask_nip.seg.nrrd"],
                [nnunet_inf_mask, "nnunet_inf.seg.nrrd"],
                [nnunet_inf_all_mask, "nnunet_inf_all.seg.nrrd"]
            ]

            for ref, name in datas:
                sitk.WriteImage(ref, os.path.join(patient_output_dir, name))


if __name__ == "__main__":
    main()