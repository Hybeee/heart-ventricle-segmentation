import cv2
import matplotlib.pyplot as plt

import json
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_images(image1, image2, output_dir, name, cmap, title, titles):
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs = axs.flatten()

    fig.suptitle(title)

    for ax in axs:
        ax.axis('off')

    axs[0].imshow(image1, cmap=cmap)
    axs[0].set_title(titles[0])
    axs[1].imshow(image2, cmap=cmap)
    axs[1].set_title(titles[1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}.png"))
    plt.close()

def main():
    output_dir = os.path.join(ROOT_DIR, "comparer_results_bbm_vs_com")
    os.makedirs(output_dir, exist_ok=True)

    patients_to_process = []
    with open(os.path.join(ROOT_DIR, "patients_to_process.txt"), 'r') as file:
        for line in file:
            line = line.strip()
            patients_to_process.append(line)
    
    dir1_name = "other_test_output_bbox_middle_old_w0"
    dir_no_nms = os.path.join(ROOT_DIR, dir1_name)

    dir2_name = "other_test_output_com_old_w0"
    dir_invalid_exclude = os.path.join(ROOT_DIR, dir2_name)

    for patient_id in patients_to_process:
        if patient_id == "patient_0025":
            continue

        dir1_pd = os.path.join(dir_no_nms, patient_id)
        dir2_pd = os.path.join(dir_invalid_exclude, patient_id)

        with open(os.path.join(dir1_pd, "result.json")) as f1, open(os.path.join(dir2_pd, "result.json")) as f2:
            dir1_result = json.load(f1)
            dir2_result = json.load(f2)
        
        dir1_threshold = dir1_result['output_data']['best_threshold']
        dir2_threshold = dir2_result['output_data']['best_threshold']

        print(f"Patient: {patient_id}")
        if dir1_threshold != dir2_threshold or 5 == 5:
            dir1_threshold_ts = dir1_result["postprocessing"][str(dir1_threshold)]["total_score"]
            dir2_threshold_ts = dir2_result["postprocessing"][str(dir2_threshold)]["total_score"]
            dir1_slice = dir1_result["preprocessing"]["middle_slice_index"]
            dir2_slice = dir2_result["preprocessing"]["middle_slice_index"]


            title = f"Thresholds: {dir1_threshold} - {dir2_threshold}\nTotal scores: {dir1_threshold_ts} - {dir2_threshold_ts}"
            title += f"\nSlice: {dir1_slice} - {dir2_slice}"

            dir1_image = cv2.imread(os.path.join(dir1_pd, "orig_result.png"))
            dir1_image = cv2.cvtColor(dir1_image, cv2.COLOR_BGR2RGB)
            dir1_polar_imag = cv2.imread(os.path.join(dir1_pd, "polar_result.png"))
            dir1_polar_imag = cv2.cvtColor(dir1_polar_imag, cv2.COLOR_BGR2RGB)

            dir2_image = cv2.imread(os.path.join(dir2_pd, "orig_result.png"))
            dir2_image = cv2.cvtColor(dir2_image, cv2.COLOR_BGR2RGB)
            dir2_polar_image = cv2.imread(os.path.join(dir2_pd, "polar_result.png"))
            dir2_polar_image = cv2.cvtColor(dir2_polar_image, cv2.COLOR_BGR2RGB)

            save_images(
                image1=dir1_image,
                image2=dir2_image,
                output_dir=output_dir,
                name=f"{patient_id}_orig",
                cmap='gray',
                title=title,
                titles=[dir1_name, dir2_name]
            )

            save_images(
                image1=dir1_polar_imag,
                image2=dir2_polar_image,
                output_dir=output_dir,
                name=f"{patient_id}_polar",
                cmap='jet',
                title=title,
                titles=[dir1_name, dir2_name]
            )

if __name__ == "__main__":
    main()