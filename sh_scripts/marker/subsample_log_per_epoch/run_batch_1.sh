#!/bin/bash

gpu_index=$1

# bash sh_scripts/marker/subsample_log_per_epoch/obesity_cae_cluster.sh $gpu_index &&
bash sh_scripts/marker/subsample_log_per_epoch/obesity_dae_cluster.sh $gpu_index &&
# bash sh_scripts/marker/subsample_log_per_epoch/obesity_sae_cluster.sh $gpu_index &&
# bash sh_scripts/marker/subsample_log_per_epoch/obesity_vae_cluster.sh $gpu_index &&
bash sh_scripts/marker/subsample_log_per_epoch/t2d_cae_cluster.sh $gpu_index &&
# bash sh_scripts/marker/subsample_log_per_epoch/t2d_dae_cluster.sh $gpu_index &&
# bash sh_scripts/marker/subsample_log_per_epoch/t2d_sae_cluster.sh $gpu_index &&
# bash sh_scripts/marker/subsample_log_per_epoch/t2d_vae_cluster.sh $gpu_index &&
# bash sh_scripts/marker/subsample_log_per_epoch/wt2d_cae_cluster.sh $gpu_index &&
# bash sh_scripts/marker/subsample_log_per_epoch/wt2d_dae_cluster.sh $gpu_index &&
bash sh_scripts/marker/subsample_log_per_epoch/wt2d_sae_cluster.sh $gpu_index # &&
# bash sh_scripts/marker/subsample_log_per_epoch/wt2d_vae_cluster.sh $gpu_index
