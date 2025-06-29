#!/bin/bash

gpu_index=$1
repetitions=10

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d marker_WT2D --ae -dm 256,128,64 -m rf --seed 1234 -r ${repetitions} --save_loss --log_per_epoch --no_clf --name "WT2D-DAE_logging"
 