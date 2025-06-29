#!/bin/bash

gpu_index=$1

CUDA_VISIBLE_DEVICES="$gpu_index" python DM.py -d abundance_Cirrhosis --vae -dm 512,8 -m svm --seed 0 -r 1 --save_loss --no_clf --name "Cirrhosis-VAE_loss"
