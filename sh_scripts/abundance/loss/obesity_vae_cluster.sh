#!/bin/bash

gpu_index=$1

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_Obesity --vae -dm 256,4 -m mlp --seed 0 -r 1 --save_loss --no_clf --name "Obesity-VAE_loss"
