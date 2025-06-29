#!/bin/bash

gpu_index=$1

bash sh_scripts/abundance/subsample_log_per_epoch/cirrhosis_cae_cluster.sh $gpu_index &&
# bash sh_scripts/abundance/subsample_log_per_epoch/cirrhosis_dae_cluster.sh $gpu_index &&
# bash sh_scripts/abundance/subsample_log_per_epoch/cirrhosis_sae_cluster.sh $gpu_index &&
# bash sh_scripts/abundance/subsample_log_per_epoch/cirrhosis_vae_cluster.sh $gpu_index &&
bash sh_scripts/abundance/subsample_log_per_epoch/colorectal_cae_cluster.sh $gpu_index &&
# bash sh_scripts/abundance/subsample_log_per_epoch/colorectal_dae_cluster.sh $gpu_index &&
# bash sh_scripts/abundance/subsample_log_per_epoch/colorectal_sae_cluster.sh $gpu_index &&
# bash sh_scripts/abundance/subsample_log_per_epoch/colorectal_vae_cluster.sh $gpu_index &&
bash sh_scripts/abundance/subsample_log_per_epoch/ibd_cae_cluster.sh $gpu_index # &&
# bash sh_scripts/abundance/subsample_log_per_epoch/ibd_dae_cluster.sh $gpu_index &&
# bash sh_scripts/abundance/subsample_log_per_epoch/ibd_sae_cluster.sh $gpu_index &&
# bash sh_scripts/abundance/subsample_log_per_epoch/ibd_vae_cluster.sh $gpu_index
