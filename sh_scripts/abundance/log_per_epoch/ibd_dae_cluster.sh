#!/bin/bash

gpu_index=$1
repetitions=10

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_IBD --ae -dm 512,256 -m mlp --seed 1234 -r ${repetitions} --save_loss --log_per_epoch --no_clf --name "IBD-DAE_logging"
