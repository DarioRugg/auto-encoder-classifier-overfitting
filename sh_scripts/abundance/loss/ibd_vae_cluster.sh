#!/bin/bash

gpu_index=$1

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_IBD --vae -dm 32,8 -m rf --seed 0 -r 1 --save_loss --no_clf --name "IBD-VAE_loss"
