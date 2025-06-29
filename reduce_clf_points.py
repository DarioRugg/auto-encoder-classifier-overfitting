import os
import pickle
import re
import shutil

points = 40
dataset_level = "abundance"  # o "marker"
experiment_type = "per_epoch"

datasets_to_work_on = ["Cirrhosis", "T2D", "IBD", "WT2D", "Obesity", "Colorectal"]

base_dir = os.path.join("results", dataset_level, experiment_type)
backup_dir = os.path.join("results", "backup", dataset_level, experiment_type)

exps_path = [
    os.path.join(base_dir, project_dir)
    for project_dir in os.listdir(base_dir)
    if any(re.match(rf"{dataset}-", project_dir) for dataset in datasets_to_work_on)
]

for exp_path in exps_path:
    print("🔍 Reducing experiments in", os.path.basename(exp_path))
    
    loss_path = os.path.join(exp_path, "losses")
    if os.path.exists(loss_path):
        mean_max_epoch = 0
        loss_files = os.listdir(loss_path)
        for file_name in loss_files:
            with open(os.path.join(loss_path, file_name), "rb") as f:
                epochs = len(pickle.load(f)["loss"])
            mean_max_epoch += epochs / len(loss_files)

        epochs_per_step = max(8, round(mean_max_epoch / points / 8) * 8)
        print(f"   ➤ Keeping 1 file every {epochs_per_step} epochs")

        repr_path = os.path.join(exp_path, "representations")
        for seed_folder_name in os.listdir(repr_path):
            seed_folder_path = os.path.join(repr_path, seed_folder_name)
            backup_seed_folder = os.path.join(
                backup_dir,
                os.path.relpath(seed_folder_path, start=base_dir)
            )
            os.makedirs(backup_seed_folder, exist_ok=True)

            for file_name in os.listdir(seed_folder_path):
                match = re.search(r"epoch_([0-9]+)", file_name)
                if not match:
                    continue

                epoch = int(match.group(1))
                src_file_path = os.path.join(seed_folder_path, file_name)

                if epoch % epochs_per_step == 0:
                    continue  # Keep file
                else:
                    dst_file_path = os.path.join(backup_seed_folder, file_name)
                    shutil.move(src_file_path, dst_file_path)
                    print(f"   🔄 Moved {file_name} → backup")
        else:
            print("   ➤ No representations to reduce.")