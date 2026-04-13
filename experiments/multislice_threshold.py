import os
import sys
import json
import math
import yaml
from copy import deepcopy

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from main import _process_one_patient

output_dir_names = {
    0: "center_output",
    1: "pos_delta_output",
    2: "neg_delta_output"
}

def _process_one_patient_multislice(patient_id: str, patient_data: dict, config: dict):
    for i in range(3):
        config["output_dir_name"] = os.path.join(ROOT_DIR, "threshold_experiment_output", patient_id, output_dir_names[i])

        center_output_dir = os.path.join(ROOT_DIR, "threshold_experiment_output", patient_id, "center_output")
        if os.path.exists(center_output_dir) and i != 0:
            center_results_path = os.path.join(center_output_dir, "results.json")
            with open(center_results_path, 'r') as f:
                center_results = json.load(f)
            
            z_data = center_results["preprocessing"]["z_data"]
            z_start = z_data["start"]
            z_middle = z_data["middle"]
            z_end = z_data["end"]

            multiplier = -1 if i % 2 == 1 else 1
            delta = max(1, math.ceil((z_end - z_start) / 10))
            new_z_middle = z_middle + multiplier * delta
            config["preprocessing"]["z_middle"] = new_z_middle

        _process_one_patient(
            patient_id=patient_id,
            patient_data=patient_data,
            config=config,
            make_output_dir=False
        )

def _process_multiple_patient_multislice(patients_to_process, patients_data, config):
    for patient_id in patients_to_process:
        patient_config = deepcopy(config)
        print(f"Processing {patient_id}")
        patient_data = patients_data[patient_id]
        _process_one_patient_multislice(
            patient_id=patient_id,
            patient_data=patient_data,
            config=patient_config
        )

def main():
    with open(os.path.join(ROOT_DIR, "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)

    with open(os.path.join(ROOT_DIR, "patients_data.json"), 'r') as f:
        patients_data = yaml.safe_load(f)

    patient_id = "patient_0001"
    patient_data = patients_data[patient_id]
    _process_one_patient_multislice(
        patient_id=patient_id,
        patient_data=patient_data,
        config=config
    )

    # patients_to_process = []
    # with open(os.path.join(ROOT_DIR, "patients_to_process.txt"), 'r') as file:
    #     for line in file:
    #         line = line.strip()
    #         patients_to_process.append(line)
    # _process_multiple_patient_multislice(
    #     patients_to_process=patients_to_process,
    #     patients_data=patients_data
    # )

if __name__ == "__main__":
    main()