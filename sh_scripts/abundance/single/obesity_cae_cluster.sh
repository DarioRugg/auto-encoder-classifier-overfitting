#!/bin/bash

gpu_index=$1
replicate=$2

for i in $(seq 0 4)
do
    echo "Execution number: $i of $to"
    echo "seed $i"
    CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_Obesity --cae -dm 4,2 -m rf --seed "$i" -r 1 --name "Obesity-CAE_single_$replicate"
done 