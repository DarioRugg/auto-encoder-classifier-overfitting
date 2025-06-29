#!/bin/bash

gpu_index=$1

bash sh_scripts/marker/subsample_log_per_epoch/ibd_sae_cluster.sh $gpu_index &&
bash sh_scripts/marker/subsample_log_per_epoch/t2d_cae_cluster.sh $gpu_index &&
bash sh_scripts/marker/subsample_log_per_epoch/cirrhosis_cae_cluster.sh $gpu_index 
