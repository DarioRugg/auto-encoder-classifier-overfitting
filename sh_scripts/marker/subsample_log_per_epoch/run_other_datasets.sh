#!/bin/bash

gpu_index=$1

bash sh_scripts/marker/subsample_log_per_epoch/colorectal_dae_cluster.sh $gpu_index &&
bash sh_scripts/marker/subsample_log_per_epoch/obesity_dae_cluster.sh $gpu_index &&
bash sh_scripts/marker/subsample_log_per_epoch/wt2d_sae_cluster.sh $gpu_index 
