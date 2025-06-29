#!/bin/bash

AE_EXP_NAME="WT2D-VAE_logging"

for SEED_DIR_PATH in results/marker/per_epoch/$AE_EXP_NAME/representations/*; do
    for FILE_PATH in $SEED_DIR_PATH/*; do
        if [[ $(basename ${FILE_PATH}) == *"train"* ]]
        then
            CUDA_VISIBLE_DEVICES=-1 python DM_classification_only.py -d marker_WT2D --vae -dm 256,16 -m svm --seed 1234 --numJobs 12 --log_per_epoch --encoded_file_path $FILE_PATH --no_trn --name $AE_EXP_NAME && wait
        fi
    done
done
 