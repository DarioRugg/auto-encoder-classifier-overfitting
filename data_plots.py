from ossaudiodev import control_names
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from sqlalchemy import column

import seaborn as sns

import random


def main():

    experiments_df = pd.read_csv("results/exp_data.csv", index_col=0, usecols=["seed", "AUC_mean"])

    bootstrap_means = []


    # for i in range(200):
    #     bootstrap_dataset = experiments_df.sample(frac=1., replace=True, random_state=random.randint(0, 100000))
    #     bootstrap_means.append(bootstrap_dataset.mean()[0])
    bootstrap_means = experiments_df["AUC_mean"]

    plt.title("Results of 100 executions", fontdict={"size":15})
    sns.distplot(bootstrap_means, hist=True, kde=True, bins=25, label="Results distribution")
    plt.axvline(x = np.quantile(bootstrap_means, q=0.025), color = 'orange', label="0.95 confidence intervals")
    plt.axvline(x = np.quantile(bootstrap_means, q=0.975), color = 'orange')
    plt.scatter(x=0.955, y=0.4, marker='o', s=100, color='r', label="Published results")
    plt.ylim((0, 20))
    plt.xlabel("AUC")
    plt.ylabel("experiments")
    plt.legend()
    plt.savefig("results/confidence_intervals.png")
    plt.close()

    print(np.abs(np.mean(bootstrap_means)-0.955)/np.std(bootstrap_means))
    print(np.abs(np.mean(bootstrap_means)-0.955), np.std(bootstrap_means))

def ci_37():

    random.seed(1234)
    
    experiments_df = pd.read_csv("results/exp_data.csv", index_col=0, usecols=["seed", "AUC_mean"])

    bootstrap_means = []
    dataset_subsampled = experiments_df.sample(n=37, replace=False, random_state=random.randint(0, 100000))
    for i in range(200):
        bootstrap_dataset = dataset_subsampled.sample(n=1, replace=True, random_state=random.randint(0, 100000))
        bootstrap_means.append(bootstrap_dataset.mean()[0])

    plt.title("Mean AUC\nfrom same number of executions", fontdict={"size":15})
    sns.distplot(bootstrap_means, hist=True, kde=True, color="grey", bins=25, label="Results distribution")
    plt.axvline(x = np.std(bootstrap_means)+np.mean(bootstrap_means), color = 'orange', linestyle='--', alpha=0.7, label="1 std")
    plt.axvline(x = np.mean(bootstrap_means)-np.std(bootstrap_means), color = 'orange', linestyle='--', alpha=0.7)
    plt.axvline(x = 2*np.std(bootstrap_means)+np.mean(bootstrap_means), color = 'darkred', linestyle='--', alpha=0.7, label="2 std")
    plt.axvline(x = np.mean(bootstrap_means)-2*np.std(bootstrap_means), color = 'darkred', linestyle='--', alpha=0.7)
    plt.scatter(x=0.884, y=0.4, marker='o', s=100, color='g', label="New results")
    plt.ylim((0, 20))
    plt.xlabel("AUC mean")
    plt.ylabel("experiments")
    plt.legend()
    plt.savefig("results/confidence_intervals.png")
    plt.close()

if __name__ == "__main__":
    ci_37()