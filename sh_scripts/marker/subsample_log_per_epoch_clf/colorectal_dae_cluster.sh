#!/bin/bash


declare -a factors=( 0.3 0.5 0.7 0.9 )
for factor in "${factors[@]}"
do
    AE_EXP_NAME="Colorectal-DAE_subsample_${factor}_logging"

    for SEED_DIR_PATH in results/marker/subsampled/$AE_EXP_NAME/representations/*; do
        for FILE_PATH in $SEED_DIR_PATH/*; do
            if [[ $(basename ${FILE_PATH}) == *"train"* ]]
            then
                CUDA_VISIBLE_DEVICES=-1 python DM_classification_only.py -d marker_Colorectal --ae -dm 512,256,128 -m mlp --seed 1234 --numJobs 12 --log_per_epoch --encoded_file_path $FILE_PATH --no_trn --name $AE_EXP_NAME && wait
            fi
        done
    done
done
 