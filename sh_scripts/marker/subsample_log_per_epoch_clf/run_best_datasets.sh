#!/bin/bash


bash sh_scripts/marker/subsample_log_per_epoch_clf/ibd_sae_cluster.sh &&
bash sh_scripts/marker/subsample_log_per_epoch_clf/t2d_cae_cluster.sh &&
bash sh_scripts/marker/subsample_log_per_epoch_clf/cirrhosis_cae_cluster.sh
