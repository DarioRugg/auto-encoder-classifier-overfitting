#!/bin/bash

gpu_index=$1
repetitions=10

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d marker_IBD --ae -dm 64 -m svm --seed 1234 -r ${repetitions} --save_loss --log_per_epoch --no_clf --name "IBD-SAE_logging"
 