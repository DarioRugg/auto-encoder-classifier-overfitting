from ossaudiodev import control_names
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff


def get_data(data_path) -> None:
    df_with_reps = pd.read_csv(data_path, usecols=['exp_num','test_auc', 'test_accuracy'])
    
    df = df_with_reps.groupby("exp_num")
    
    print(" -", data_path.split("/")[1].replace("clf_results_", "").replace("_", " "))
    print(" -- one sample")
    print(" ---> AUC:      {} ({})".format(df.mean()["test_auc"].iloc[0].round(3), df.std()["test_auc"].iloc[0].round(3)))
    print(" ---> Accuracy: {} ({})".format(df.mean()["test_accuracy"].iloc[0].round(3), df.std()["test_accuracy"].iloc[0].round(3)))
    print(" -- metrics")
    print(" ---> AUC:      {} ({})".format(df.mean()["test_auc"].mean().round(3), df.std()["test_auc"].mean().round(3)))
    print(" ---> Accuracy: {} ({})".format(df.mean()["test_accuracy"].mean().round(3), df.std()["test_accuracy"].mean().round(3)))
    print()
        
        
if __name__ == "__main__":
    for datapath in os.listdir("results/"):
        if "clf_results" in datapath:
            get_data(os.path.join("results/", datapath))