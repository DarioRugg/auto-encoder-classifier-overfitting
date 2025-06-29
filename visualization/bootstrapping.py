from ossaudiodev import control_names
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from sqlalchemy import column

import seaborn as sns

import random


def necessary_sample_size(confidence_interval, mean, std, error):
    pass

def main():

    experiments_df = pd.read_csv("results/exp_data.csv", index_col=0, usecols=["seed", "AUC_mean"])

    bootstrap_means = []

    random.seed(1234)

    for i in range(200):
        bootstrap_dataset = experiments_df.sample(frac=1., replace=True, random_state=random.randint(0, 100000))
        bootstrap_means.append(bootstrap_dataset.mean()[0])

    bins = 30
    plt.hist(bootstrap_means, bins, label="complete")
    plt.title("Histogram", fontdict={"size":15})
    plt.xlabel("AUC")
    plt.ylabel("experiments")
    plt.savefig("results/means_hist.png")

    results_df = pd.DataFrame()
    results_df["complete"] = bootstrap_means
    for num_samples in list(range(10, 40, 10)) + list(range(40, len(experiments_df), 20)):
        means = []
        dataset_subsampled = experiments_df.sample(n=num_samples, replace=False, random_state=random.randint(0, 100000))
        for i in range(200):
            bootstrap_dataset = dataset_subsampled.sample(n=1, replace=True, random_state=random.randint(0, 100000))
            means.append(bootstrap_dataset.mean()[0])
        results_df[str(num_samples) + " samples"] = means

        
        plt.hist(means, bins, alpha=0.3, label='{} samples'.format(num_samples))
            
    
    plt.legend()
    plt.savefig("results/sub_means_hist.png")
    plt.close()

    for column, data in results_df.items():
        sns.kdeplot(data)

    plt.savefig("results/test.png")
    plt.close()

    print(results_df.describe())


    print(results_df.mean())

    average_loss = []
    results_df = pd.DataFrame()
    results_df["complete"] = bootstrap_means
    for num_samples in range(4, len(experiments_df), 2):
        means = []
        for i in range(200):
            bootstrap_dataset = experiments_df.sample(n=num_samples, replace=True, random_state=random.randint(0, 100000))
            means.append(bootstrap_dataset.mean()[0])
        results_df[str(num_samples) + " samples"] = means
        
        sns.kdeplot(results_df["complete"], shade=True)
        sns.kdeplot(results_df[str(num_samples) + " samples"], shade=True)
        plt.title("Means distribution when subsampling", fontdict={"size":15})
        plt.xlabel("AUC mean")
        plt.ylabel("experiments neans")
        
        plt.xlim([0.85, 0.90])
        plt.savefig("results/imgs/test_{}.png".format(num_samples))
        plt.close()

        average_loss.append(np.abs(np.mean(means) - np.mean(bootstrap_means)))

    sns.kdeplot(results_df["complete"], shade=True)
    for num_samples in range(40, 79, 12):
        sns.kdeplot(results_df[str(num_samples) + " samples"], shade=True)
    
    
    plt.title("Candidate subsampling size", fontdict={"size":15})
    plt.xlabel("AUC mean")
    plt.ylabel("experiments")
    plt.savefig("results/imgs/summary.png".format(num_samples))
    plt.close()
if __name__ == "__main__":
    main()