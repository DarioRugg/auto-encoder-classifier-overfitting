#!/bin/bash

gpu_index=$1
repetitions=10

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_Cirrhosis --ae -dm 32 -m svm --seed 1234 -r ${repetitions} --save_loss --log_per_epoch --no_clf --name "Cirrhosis-SAE_logging"
