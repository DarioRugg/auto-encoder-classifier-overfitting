#!/bin/bash

gpu_index=$1
from=$2
to=$3

for i in $(seq "$from" "$to")
do
    echo "Execution number: $i of $to"
    echo "seed $i"
    CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_Cirrhosis --ae -dm 1024,512 -m mlp --numJobs 8 --seed "$i" --name "Cirrhosis-DAE"
done 