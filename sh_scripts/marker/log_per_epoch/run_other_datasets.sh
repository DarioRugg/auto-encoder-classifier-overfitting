#!/bin/bash

gpu_index=$1

sh sh_scripts/marker/log_per_epoch/colorectal_dae_cluster.sh $gpu_index &&
sh sh_scripts/marker/log_per_epoch/obesity_dae_cluster.sh $gpu_index &&
sh sh_scripts/marker/log_per_epoch/wt2d_sae_cluster.sh $gpu_index # &&