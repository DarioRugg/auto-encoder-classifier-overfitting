#!/bin/bash

gpu_index=$1

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_Colorectal --cae -dm 8,4 -m rf --seed 0 -r 1 --save_loss --no_clf --name "Colorectal-CAE_loss"
