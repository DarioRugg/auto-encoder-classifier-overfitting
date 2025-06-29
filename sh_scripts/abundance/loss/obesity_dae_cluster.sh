#!/bin/bash

gpu_index=$1

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_Obesity --ae -dm 1024,512 -m mlp --seed 0 -r 1 --save_loss --no_clf --name "Obesity-DAE_loss"
