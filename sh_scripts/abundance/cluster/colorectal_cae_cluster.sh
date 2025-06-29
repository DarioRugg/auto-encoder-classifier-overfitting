#!/bin/bash

gpu_index=$1
from=$2
to=$3

for i in $(seq "$from" "$to")
do
    echo "Execution number: $i of $to"
    echo "seed $i"
    CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_Colorectal --cae -dm 8,4 -m rf --numJobs 8 --seed "$i" --name "Colorectal-CAE"
done 