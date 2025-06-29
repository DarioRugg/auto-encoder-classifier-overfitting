#!/bin/bash

# Config
LEGACY_IMAGE="rugg/deepmicro:legacy"
WORKDIR="/workspaces/DeepMicro"
PROJECT_DIR="/srv/nfs-data/ruggeri/DeepMicro_main"
DATA_DIR="/srv/nfs-data/ruggeri/datasets/IBD"

# List of scripts to run
SCRIPTS=(
    "sh_scripts/abundance/log_per_epoch_clf/ibd_cae_cluster.sh"
    "sh_scripts/abundance/log_per_epoch_clf/t2d_cae_cluster.sh"
    "sh_scripts/abundance/log_per_epoch_clf/cirrhosis_cae_cluster.sh"
    "sh_scripts/abundance/log_per_epoch_clf/obesity_cae_cluster.sh"
    "sh_scripts/abundance/log_per_epoch_clf/colorectal_cae_cluster.sh"
    "sh_scripts/abundance/log_per_epoch_clf/wt2d_cae_cluster.sh"
    "sh_scripts/marker/log_per_epoch_clf/ibd_sae_cluster.sh"
    "sh_scripts/marker/log_per_epoch_clf/t2d_cae_cluster.sh"
    "sh_scripts/marker/log_per_epoch_clf/cirrhosis_cae_cluster.sh"
    "sh_scripts/marker/log_per_epoch_clf/colorectal_dae_cluster.sh"
    "sh_scripts/marker/log_per_epoch_clf/obesity_dae_cluster.sh"
    "sh_scripts/marker/log_per_epoch_clf/wt2d_sae_cluster.sh"
)

mkdir -p logs

for SCRIPT in "${SCRIPTS[@]}"; do
    SCRIPT_NAME=$(basename "$SCRIPT")
    LOGFILE="logs/${SCRIPT_NAME%.sh}_clf.log"

    echo "▶️ Running $SCRIPT..."
    
    docker run --rm \
                --name "legacy_run_${SCRIPT_NAME%.sh}" \
                --gpus all \
                -v "$PROJECT_DIR:$WORKDIR" \
                -v "$DATA_DIR:$WORKDIR/data" \
                -w "$WORKDIR" \
                "$LEGACY_IMAGE" \
                bash "$SCRIPT" > "$LOGFILE" 2>&1

    echo "✅ Finished $SCRIPT → logged to $LOGFILE"
done

echo "🎉 All scripts completed."
