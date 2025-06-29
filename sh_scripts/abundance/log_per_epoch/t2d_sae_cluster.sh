#!/bin/bash

gpu_index=$1
repetitions=10

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_T2D --ae -dm 64 -m svm --seed 1234 -r ${repetitions} --save_loss --log_per_epoch --no_clf --name "T2D-SAE_logging"
