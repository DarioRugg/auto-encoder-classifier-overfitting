#!/bin/bash

gpu_index=$1
repetitions=10

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_Cirrhosis --cae -dm 16,8,4 -m rf --seed 1234 -r ${repetitions} --save_loss --log_per_epoch --no_clf --name "Cirrhosis-CAE_logging"
