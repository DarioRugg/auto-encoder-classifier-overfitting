# Investigating the Impact of Autoencoder Overfitting on Downstream Classification in HDLSS Context

This repository contains the official implementation of the experiments from the paper:

**“Investigating the Impact of Autoencoder Overfitting on Downstream Classification in High-Dimensional Low-Sample Size Context”**
by *Dario Ruggeri and László Vidács*

## Overview

This work explores the role of autoencoder (AE) overfitting on downstream classification tasks in microbiome data characterized by high dimensionality and low sample sizes (HDLSS). We systematically compare AE representations from models trained to overfitting vs. early stopping and evaluate their effect on classifier performance.

The pipeline is based on the DeepMicro framework and extends it with large-scale experimental clusters and per-epoch classifier evaluations.

---

## Dataset Setup

The datasets used in this study are derived from those in the [DeepMicro repository](https://github.com/minoh0201/DeepMicro/tree/master/data).

### Steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/DarioRugg/auto-encoder-classifier-overfitting
   cd auto-encoder-classifier-overfitting
   ```

2. Download and unzip datasets from [DeepMicro/data](https://github.com/minoh0201/DeepMicro/tree/master/data), and place them in the `data` folder:

   ```bash
   mkdir -p data
   # Manually download & unzip datasets into ./data/
   ```

---

## Docker Environment

All experiments must be run inside a Docker container to ensure reproducibility.

### Build the Docker image:

```bash
docker build -f legacy.Dockerfile -t ae-overfit-env .
```

### Run the container:

```bash
docker run -it --gpus all --mount type=bind,source=$(pwd),target=/workspace ae-overfit-env
cd /workspace
```

---

## Running the Experiments

All experiment scripts are located in the `sh_scripts/` folder.

### Cluster Experiments (Main Analysis)

To replicate the cluster of 60 experiments per dataset-AE combination (as reported in the paper), use the following:

```bash
# Format
sh sh_scripts/<abundance|marker>/<dataset>_<AE>_cluster.sh <gpu_index> <from> <to>

# Example: Run shallow AE on marker profile of IBD, seeds 0–59 on GPU 0
sh sh_scripts/marker/IBD_shallow_cluster.sh 0 0 59
```

These scripts:

* Use best-performing hyperparameters for AE and the best classifier model.
* Still perform k-fold CV for classifier tuning.
* Replicate the results in the paper when using 0–59 seeds ranges.

---

## AE Training Monitoring + Classifier Performance per Epoch

This analysis monitors classifier AUC every few epochs during AE training.

1. **Train AE with logging every 4 epochs**
   (Creates latent representations used for classification):

```bash
# Format
sh sh_scripts/<abundance|marker>/log_per_epoch/<dataset>_<AE>.sh <gpu_index>

# Example: shallow AE on IBD strain-level, using GPU 0
sh sh_scripts/marker/log_per_epoch/IBD_shallow.sh 0
```

2. **(Optional)** Reduce number of representations to 40 to speed up classification:
Useful when working with dataset–AE combinations that require long training schedules and generate many intermediate representations.

```bash
python reduce_clf_points.py
```

3. **Run classifier on saved representations**
   (**No arguments required** — runs on default 10 seeds):

```bash
# Format
sh sh_scripts/<abundance|marker>/log_per_epoch_clf/<dataset>_<AE>.sh

# Example
sh sh_scripts/marker/log_per_epoch_clf/IBD_shallow.sh
```
