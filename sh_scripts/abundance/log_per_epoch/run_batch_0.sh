#!/bin/bash

gpu_index=$1

sh sh_scripts/abundance/log_per_epoch/cirrhosis_cae_cluster.sh $gpu_index &&
# sh sh_scripts/abundance/log_per_epoch/cirrhosis_dae_cluster.sh $gpu_index &&
# sh sh_scripts/abundance/log_per_epoch/cirrhosis_sae_cluster.sh $gpu_index &&
# sh sh_scripts/abundance/log_per_epoch/cirrhosis_vae_cluster.sh $gpu_index &&
sh sh_scripts/abundance/log_per_epoch/colorectal_cae_cluster.sh $gpu_index &&
# sh sh_scripts/abundance/log_per_epoch/colorectal_dae_cluster.sh $gpu_index &&
# sh sh_scripts/abundance/log_per_epoch/colorectal_sae_cluster.sh $gpu_index &&
# sh sh_scripts/abundance/log_per_epoch/colorectal_vae_cluster.sh $gpu_index &&
sh sh_scripts/abundance/log_per_epoch/ibd_cae_cluster.sh $gpu_index # &&
# sh sh_scripts/abundance/log_per_epoch/ibd_dae_cluster.sh $gpu_index &&
# sh sh_scripts/abundance/log_per_epoch/ibd_sae_cluster.sh $gpu_index # &&
# sh sh_scripts/abundance/log_per_epoch/ibd_vae_cluster.sh $gpu_index
