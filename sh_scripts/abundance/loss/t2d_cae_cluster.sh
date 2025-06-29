#!/bin/bash

gpu_index=$1

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_T2D --cae -dm 4,2,1 -m rf --seed 0 -r 1 --save_loss --no_clf --name "T2D-CAE_loss"
