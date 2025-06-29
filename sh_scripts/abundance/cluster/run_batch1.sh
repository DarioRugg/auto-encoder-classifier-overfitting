#!/bin/bash

sh sh_scripts/obesity_cae_cluster.sh 2 0 59 &&
sh sh_scripts/obesity_dae_cluster.sh 2 26 59 &&
sh sh_scripts/obesity_sae_cluster.sh 2 0 59 &&
sh sh_scripts/obesity_vae_cluster.sh 2 0 59 &&
sh sh_scripts/t2d_cae_cluster.sh 2 0 59 &&
sh sh_scripts/t2d_dae_cluster.sh 2 0 59 &&
sh sh_scripts/t2d_sae_cluster.sh 2 0 59 &&
sh sh_scripts/t2d_vae_cluster.sh 2 0 59 &&
sh sh_scripts/wt2d_cae_cluster.sh 2 0 59 &&
sh sh_scripts/wt2d_dae_cluster.sh 2 0 59 &&
sh sh_scripts/wt2d_sae_cluster.sh 2 0 59 &&
sh sh_scripts/wt2d_vae_cluster.sh 2 0 59
