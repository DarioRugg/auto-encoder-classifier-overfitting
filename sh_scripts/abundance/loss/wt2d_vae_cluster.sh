#!/bin/bash

gpu_index=$1

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_WT2D --vae -dm 64,8 -m rf --seed 0 -r 1 --save_loss --no_clf --name "WT2D-VAE_loss"
