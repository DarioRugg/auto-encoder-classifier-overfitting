#!/bin/bash

sh sh_scripts/wt2d_dae_cluster.sh 0 0 59 &&
sh sh_scripts/wt2d_sae_cluster.sh 0 0 59 &&
sh sh_scripts/wt2d_vae_cluster.sh 0 0 59
