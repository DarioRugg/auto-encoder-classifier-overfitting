#!/bin/bash

gpu_index=$1

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_WT2D --ae -dm 256 -m svm --seed 0 -r 1 --save_loss --no_clf --name "WT2D-SAE_loss"
