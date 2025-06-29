#!/bin/bash

gpu_index=$1

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_Cirrhosis --cae -dm 16,8,4 -m rf --seed 0 -r 1 --save_loss --no_clf --name "Cirrhosis-CAE_loss"
