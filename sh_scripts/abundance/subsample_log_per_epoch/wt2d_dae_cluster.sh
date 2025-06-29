#!/bin/bash

gpu_index=$1
repetitions=10


declare -a factors=( 0.3 0.5 0.7 0.9 )
for factor in "${factors[@]}"
do
	CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_WT2D --ae -dm 1024,512 -m svm --seed 1234 -r ${repetitions} --save_loss --subsample --subsample_factor ${factor} --log_per_epoch --no_clf --name "WT2D-DAE_${factor}_logging"
done