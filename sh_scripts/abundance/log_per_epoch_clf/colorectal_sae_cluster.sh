#!/bin/bash

AE_EXP_NAME="Colorectal-SAE_logging"

for SEED_DIR_PATH in results/abundance/per_epoch/$AE_EXP_NAME/representations/*; do
    for FILE_PATH in $SEED_DIR_PATH/*; do
        if [[ $(basename ${FILE_PATH}) == *"train"* ]]
        then
            CUDA_VISIBLE_DEVICES=-1 python DM_classification_only.py -d abundance_Colorectal --ae -dm 256 -m svm --no_trn --seed 1234 --numJobs 12 --log_per_epoch --encoded_file_path $FILE_PATH --name $AE_EXP_NAME && wait
        fi
    done
done
