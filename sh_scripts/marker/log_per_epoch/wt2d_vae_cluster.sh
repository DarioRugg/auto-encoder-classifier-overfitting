#!/bin/bash

gpu_index=$1
repetitions=10

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d marker_WT2D --vae -dm 256,16 -m svm --seed 1234 -r ${repetitions} --save_loss --log_per_epoch --no_clf --name "WT2D-VAE_logging"
 