from ossaudiodev import control_names
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt


def main():
    results_dir = "results.csv"

    results_df = pd.read_csv(results_dir, header=None)
    results_df.columns = ["seed", "train", "test", "val"]
    
    print(results_df.mean(), results_df.std())
    
    plots(results_df, "results/results_summary")
    
def plots(df, save_dir):
    # AUC mean
    col_name = "train"
    plt.hist(df[col_name], color="olive")
    plt.title("{} distribution".format(col_name.capitalize().replace("_", " ")))
    plt.text(0.775, 9.5, 'mean: {:.3}\nstd: {:.4}'.format(df[col_name].mean(), df[col_name].std()))
    plt.xlabel(col_name.capitalize().replace("_", " "))
    plt.ylabel("number of samles")
    plt.savefig(os.path.join(save_dir, "{}_histogram.png".format(col_name)))
    plt.close()

    # AUC std
    col_name = "test"
    plt.hist(df[col_name], color="orange")
    plt.text(0.14, 5.5, 'mean: {:.3}\nstd: {:.4}'.format(df[col_name].mean(), df[col_name].std()))
    plt.title("{} distribution".format(col_name.capitalize().replace("_", " ")))
    plt.xlabel(col_name.capitalize().replace("_", " "))
    plt.ylabel("number of samles")
    plt.savefig(os.path.join(save_dir, "{}_histogram.png".format(col_name)))
    plt.close()

    # Accuracy mean
    col_name = "val"
    plt.hist(df[col_name], color="olive")
    plt.title("{} distribution".format(col_name.capitalize().replace("_", " ")))
    plt.text(0.81, 35, 'mean: {:.3}\nstd: {:.4}'.format(df[col_name].mean(), df[col_name].std()))
    plt.xlabel(col_name.capitalize().replace("_", " "))
    plt.ylabel("number of samles")
    plt.savefig(os.path.join(save_dir, "{}_histogram.png".format(col_name)))
    plt.close()


if __name__ == "__main__":
    main()