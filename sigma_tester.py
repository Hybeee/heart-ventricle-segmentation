import os
import yaml
import copy

from main import _process_multiple_patients

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    with open(os.path.join(ROOT_DIR, "config.yaml")) as f:
        orig_config = yaml.safe_load(f)
    
    with open(os.path.join(ROOT_DIR, "patients_data.json"), 'r') as f:
        patients_data = yaml.safe_load(f)

    sigmas = [1.2, 1.8, 2.16, 2.4, 2.64, 3.0, 3.6, 4.8]

    patients_to_process = []
    with open(os.path.join(ROOT_DIR, "patients_to_process.txt"), 'r') as file:
        for line in file:
            line = line.strip()
            patients_to_process.append(line)

    for sigma in sigmas:
        output_dir = os.path.join("sigma_exp", str(sigma))

        config = copy.deepcopy(orig_config)

        config["output_dir_name"] = output_dir
        config["preprocessing"]["sigma"] = sigma

        print(f"Sigma: {sigma}")

        _process_multiple_patients(
            patients_to_process=patients_to_process,
            patients_data=patients_data,
            config=config
        )

if __name__ == "__main__":
    main()
