#!/bin/bash

gpu_index=$1
repetitions=10

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d marker_Colorectal --ae -dm 512,256,128 -m mlp --seed 1234 -r ${repetitions} --save_loss --log_per_epoch --no_clf --name "Colorectal-DAE_logging"
 