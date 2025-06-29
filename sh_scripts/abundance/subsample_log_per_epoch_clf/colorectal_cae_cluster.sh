#!/bin/bash


declare -a factors=( 0.3 0.5 0.7 0.9 )
for factor in "${factors[@]}"
do
	AE_EXP_NAME="Colorectal-CAE_subsample_${factor}_logging"

	for SEED_DIR_PATH in results/abundance/subsampled/$AE_EXP_NAME/representations/*; do
	    for FILE_PATH in $SEED_DIR_PATH/*; do
            if [[ $(basename ${FILE_PATH}) == *"train"* ]]
            then
                CUDA_VISIBLE_DEVICES=-1 python DM_classification_only.py -d abundance_Colorectal --cae -dm 8,4 -m rf --no_trn --seed 1234 --numJobs 12 --log_per_epoch --encoded_file_path $FILE_PATH --name $AE_EXP_NAME && wait
            fi
        done
    done
done
