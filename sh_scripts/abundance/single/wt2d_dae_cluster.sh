#!/bin/bash

gpu_index=$1
replicate=$2

for i in $(seq 0 4)
do
    echo "Execution number: $i of $to"
    echo "seed $i"
    CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_WT2D --ae -dm 1024,512 -m svm --seed "$i" -r 1 --name "WT2D-DAE_single_$replicate"
done 