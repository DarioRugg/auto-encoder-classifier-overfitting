#!/bin/bash

gpu_index=$1

bash sh_scripts/abundance/subsample_log_per_epoch/colorectal_cae_cluster.sh $gpu_index &&
bash sh_scripts/abundance/subsample_log_per_epoch/obesity_cae_cluster.sh $gpu_index &&
bash sh_scripts/abundance/subsample_log_per_epoch/wt2d_cae_cluster.sh $gpu_index