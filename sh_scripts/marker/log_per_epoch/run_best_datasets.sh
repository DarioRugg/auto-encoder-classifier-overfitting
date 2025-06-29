#!/bin/bash

gpu_index=$1

sh sh_scripts/marker/log_per_epoch/ibd_sae_cluster.sh $gpu_index &&
sh sh_scripts/marker/log_per_epoch/t2d_cae_cluster.sh $gpu_index &&
sh sh_scripts/marker/log_per_epoch/cirrhosis_cae_cluster.sh $gpu_index
