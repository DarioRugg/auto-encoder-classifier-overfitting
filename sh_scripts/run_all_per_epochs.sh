#!/bin/bash

# Config
PROFILE="abundance"

LEGACY_IMAGE="rugg/deepmicro:legacy"
WORKDIR="/workspaces/DeepMicro"
SCRIPT_DIR="sh_scripts/$PROFILE/log_per_epoch"
CONTAINER_PREFIX="deepmicro_legacy"

DATA_DIR="/srv/nfs-data/ruggeri/datasets/IBD/"
PROJECT_DIR="/srv/nfs-data/ruggeri/DeepMicro_main/"

# Get GPU list
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <GPU_IDs...> (e.g., $0 0 2 3)"
    exit 1
fi

GPUS=("$@")
NUM_GPUS=${#GPUS[@]}

# List of scripts
SCRIPTS=(${SCRIPT_DIR}/*.sh)
NUM_SCRIPTS=${#SCRIPTS[@]}

if [ "$NUM_SCRIPTS" -eq 0 ]; then
    echo "No scripts found in $SCRIPT_DIR"
    exit 1
fi

echo "Found $NUM_SCRIPTS scripts. Distributing across $NUM_GPUS GPU(s)..."

# Track PIDs and container names
PIDS=()
CONTAINERS=()

# Function to clean up containers and background jobs
cleanup() {
    echo "Caught Ctrl+C — stopping containers and killing background jobs..."
    for CONTAINER in "${CONTAINERS[@]}"; do
        docker stop "$CONTAINER" > /dev/null 2>&1 || true
    done
    for PID in "${PIDS[@]}"; do
        kill "$PID" > /dev/null 2>&1 || true
    done
    exit 1
}

trap cleanup SIGINT

# Round-robin script assignment per GPU
for i in "${!GPUS[@]}"; do
    GPU_ID=${GPUS[$i]}

    (
        j=0
        for ((index=i; index<NUM_SCRIPTS; index+=NUM_GPUS)); do
            SCRIPT_PATH="$WORKDIR/${SCRIPTS[$index]}"
            SCRIPT_NAME=$(basename "$SCRIPT_PATH")
            CONTAINER_NAME="${CONTAINER_PREFIX}_${GPU_ID}_${j}"

            echo "▶️ [GPU $GPU_ID] Running $SCRIPT_NAME in container $CONTAINER_NAME..."

            docker run --rm \
                --gpus device=$GPU_ID \
                --name "$CONTAINER_NAME" \
                -v $PROJECT_DIR:$WORKDIR \
                -v $DATA_DIR:$WORKDIR/data/ \
                -w $WORKDIR \
                $LEGACY_IMAGE \
                bash "$SCRIPT_PATH" "$GPU_ID" > logs/${PROFILE}_$(basename "$SCRIPT_PATH" .sh)_gpu${GPU_ID}.log 2>&1


            CONTAINERS+=("$CONTAINER_NAME")
            PIDS+=($!)

            echo "✅ [GPU $GPU_ID] Done with $SCRIPT_NAME"
            ((j++))
        done
    ) &
done

wait
echo "🎉 All scripts have finished running inside legacy containers."
