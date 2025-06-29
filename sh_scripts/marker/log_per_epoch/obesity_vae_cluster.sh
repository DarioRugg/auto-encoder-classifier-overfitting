#!/bin/bash

gpu_index=$1
repetitions=10

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d marker_Obesity --vae -dm 512,8 -m rf --seed 1234 -r ${repetitions} --save_loss --log_per_epoch --no_clf --name "Obesity-VAE_logging"
 