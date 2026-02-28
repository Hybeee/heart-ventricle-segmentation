import cv2
import matplotlib.pyplot as plt

import json
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_images(image1, image2, output_dir, name, cmap):
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs = axs.flatten()

    for ax in axs:
        ax.axis('off')

    axs[0].imshow(image1, cmap=cmap)
    axs[0].set_title("No NMS")
    axs[1].imshow(image2, cmap=cmap)
    axs[1].set_title("NMS")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}.png"))
    plt.close()

def main():
    output_dir = os.path.join(ROOT_DIR, "test")

    patients_to_process = []
    with open(os.path.join(ROOT_DIR, "patients_to_process.txt"), 'r') as file:
        for line in file:
            line = line.strip()
            patients_to_process.append(line)

    dir_no_nms = os.path.join(ROOT_DIR, "output_no_nms")
    dir_invalid_exclude = os.path.join(ROOT_DIR, "output_invalid_exclude")

    for patient_id in patients_to_process:
        if patient_id == "patient_0025":
            continue

        no_nms_pd = os.path.join(dir_no_nms, patient_id)
        invalid_exclude_pd = os.path.join(dir_invalid_exclude, patient_id)

        with open(os.path.join(no_nms_pd, "result.json")) as f1, open(os.path.join(invalid_exclude_pd, "result.json")) as f2:
            no_nms_result = json.load(f1)
            invalid_exclude_result = json.load(f2)
        
        no_nms_threshold = no_nms_result['output_data']['best_threshold']
        invalid_exclude_threshold = invalid_exclude_result['output_data']['best_threshold']

        print(f"Patient: {patient_id}")
        print(f"\tNo NMS - Invalid area excluded")
        print(f"\t{no_nms_threshold} - {invalid_exclude_threshold}")
        if no_nms_threshold != invalid_exclude_threshold:
            no_nms_threshold_dds = no_nms_result["postprocessing"][str(no_nms_threshold)]["distance_decider_score"]
            invalid_exclude_threshold_dds = invalid_exclude_result["postprocessing"][str(invalid_exclude_threshold)]["distance_decider_score"]
            
            print(f"\t{no_nms_threshold_dds} - {invalid_exclude_threshold_dds}")

            nn_image = cv2.imread(os.path.join(no_nms_pd, "orig_result.png"))
            nn_image = cv2.cvtColor(nn_image, cv2.COLOR_BGR2RGB)
            nn_polar_image = cv2.imread(os.path.join(no_nms_pd, "polar_result.png"))
            nn_polar_image = cv2.cvtColor(nn_polar_image, cv2.COLOR_BGR2RGB)

            ie_image = cv2.imread(os.path.join(invalid_exclude_pd, "orig_result.png"))
            ie_image = cv2.cvtColor(ie_image, cv2.COLOR_BGR2RGB)
            ie_polar_image = cv2.imread(os.path.join(invalid_exclude_pd, "polar_result.png"))
            ie_polar_image = cv2.cvtColor(ie_polar_image, cv2.COLOR_BGR2RGB)

            save_images(
                image1=nn_image,
                image2=ie_image,
                output_dir=output_dir,
                name=f"{patient_id}_orig",
                cmap='gray'
            )

            save_images(
                image1=nn_polar_image,
                image2=ie_polar_image,
                output_dir=output_dir,
                name=f"{patient_id}_polar",
                cmap='jet'
            )

if __name__ == "__main__":
    main()