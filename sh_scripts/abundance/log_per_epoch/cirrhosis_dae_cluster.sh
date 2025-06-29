#!/bin/bash

gpu_index=$1
repetitions=10

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_Cirrhosis --ae -dm 1024,512 -m mlp --seed 1234 -r ${repetitions} --save_loss --log_per_epoch --no_clf --name "Cirrhosis-DAE_logging"
