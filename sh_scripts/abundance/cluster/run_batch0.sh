#!/bin/bash

sh sh_scripts/cirrhosis_cae_cluster.sh 1 0 59 &&
sh sh_scripts/cirrhosis_dae_cluster.sh 1 27 59 &&
sh sh_scripts/cirrhosis_sae_cluster.sh 1 0 59 &&
sh sh_scripts/cirrhosis_vae_cluster.sh 1 0 59 &&
sh sh_scripts/colorectal_cae_cluster.sh 1 0 59 &&
sh sh_scripts/colorectal_dae_cluster.sh 1 0 59 &&
sh sh_scripts/colorectal_sae_cluster.sh 1 0 59 &&
sh sh_scripts/colorectal_vae_cluster.sh 1 0 59 &&
sh sh_scripts/ibd_cae_cluster.sh 1 0 59 &&
sh sh_scripts/ibd_dae_cluster.sh 1 0 59 &&
sh sh_scripts/ibd_sae_cluster.sh 1 0 59 &&
sh sh_scripts/ibd_vae_cluster.sh 1 0 59
