#!/bin/bash

gpu_index=$1
from=$2
to=$3

for i in $(seq "$from" "$to")
do
    echo "Execution number: $i of $to"
    echo "seed $i"
    CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d marker_WT2D --vae -dm 256,16 -m svm --numJobs 8 --seed "$i" --name "WT2D-VAE"
done 