import subprocess
import os
import json

# Base directory paths
# base_dir = "/home/kev/instant-ngp/data/unit"
# output_base_dir = "/home/kev/instant-ngp/data/unit"

base_dir = "/home/kev/instant-ngp/data/unit_attacked"
output_base_dir = "/home/kev/instant-ngp/data/unit_attacked"
os.makedirs(output_base_dir, exist_ok=True)  # Create output directory if not exists

# Parameters
n_steps = 1000  # Number of training steps
marching_cubes_res = 512  # Mesh resolution
density_thresh = 2.5  # Density threshold

# Function to train and save mesh for a dataset
def train_and_save_mesh(dataset_path, dataset_name):
    output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    output_mesh = os.path.join(output_dir, f"{dataset_name}.obj")
    log_file = os.path.join(output_dir, "log.txt")

    # Command for training and saving the mesh
    command = [
        "python", "/home/kev/instant-ngp/scripts/run.py",
        "--scene", dataset_path,
        "--n_steps", str(n_steps),
        "--marching_cubes_res", str(marching_cubes_res),
        "--save_mesh", output_mesh,
        "--marching_cubes_density_thresh", str(density_thresh),
        "--nerf_compatibility",
        "--train"
    ]

    try:
        print(f"Processing dataset: {dataset_name}")
        with open(log_file, "w") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True
            )
            process.wait()

        if process.returncode == 0:
            print(f"Mesh saved for dataset: {dataset_name}")
        else:
            print(f"Error occurred for dataset: {dataset_name}. Check log: {log_file}")

    except Exception as e:
        print(f"Unexpected error for dataset {dataset_name}: {e}")

# Main loop to process all datasets
def process_datasets():
    datasets = os.listdir(base_dir)
    skipped_datasets = []

    for dataset in datasets:
        dataset_path = os.path.join(base_dir, dataset)

        # Check if transforms.json exists
        transforms_file = os.path.join(dataset_path, "transforms.json")
        if not os.path.exists(transforms_file):
            print(f"Skipping dataset {dataset}: Missing transforms.json")
            skipped_datasets.append(dataset)
            continue

        # Process dataset
        train_and_save_mesh(dataset_path, dataset)

    # Log skipped datasets
    skipped_log = os.path.join(output_base_dir, "skipped_datasets.txt")
    with open(skipped_log, "w") as f:
        f.write("\n".join(skipped_datasets))
    print(f"Skipped datasets logged in: {skipped_log}")

# Execute the processing
process_datasets()