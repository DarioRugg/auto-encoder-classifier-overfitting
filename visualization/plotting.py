from ossaudiodev import control_names
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns


# using the style for the plot
plt.style.use('seaborn')


class PlotterClass:
    def __init__(self, ae_results_path, clf_results_path, dataset_name:str, paper_metrics, save_dir) -> None:
        if ae_results_path:
            self.ae_df_with_reps = pd.read_csv(ae_results_path)
            self.ae_df = self.ae_df_with_reps.drop(columns="seed").groupby("exp_num").mean()
        
        self.clf_df_with_reps = pd.read_csv(clf_results_path)
        self.clf_df = self.clf_df_with_reps.drop(columns="seed").groupby("exp_num").mean()
        
        self.metrics = paper_metrics
        self.dataset_name = dataset_name

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
    def plot_density(self, model, col_name, color):
        assert model in ("ae", "clf")

        df = self.ae_df if model == "ae" else self.clf_df
        
        fig, hist_ax = plt.subplots()

        sns.distplot(df[col_name], kde=False, bins=25, color=color, label="Repetitions", ax=hist_ax)
        plt.ylabel("Counts")
        plt.xlabel(self.prettify_label(col_name))
        # if model == "ae":
        #     plt.xlabel("Auto-Encoder reconstruction {}".format(self.prettify_label(col_name).split()[-1].lower()))
        # else:
        #     plt.xlabel("Classifier {} score".format(self.prettify_label(col_name).split()[-1]))
        # hist_ax.set_facecolor('lavender')
        

        dens_ax = hist_ax.twinx()
        sns.distplot(df[col_name], rug=True, hist=False, color=color, ax=dens_ax)
        plt.ylabel("Density")
        
        dens_ax.grid(False)
        
        plt.title("Distribution Plot")
        
        results_added = self.add_autors_results(col_name)
        if results_added:
            plt.axvline(np.mean(df[col_name]), color="green", linestyle="dashdot", label="Mean")
            plt.axvline(np.mean(df[col_name]) - np.std(df[col_name]), color="darkorange", linestyle="dashdot", label="1 STD")
            plt.axvline(np.mean(df[col_name]) + np.std(df[col_name]), color="darkorange", linestyle="dashdot")
            plt.axvline(np.mean(df[col_name]) - 2 * np.std(df[col_name]), color="darkred", linestyle="dashdot", label="2 STD")
            plt.axvline(np.mean(df[col_name]) + 2 * np.std(df[col_name]), color="darkred", linestyle="dashdot")
        plt.legend(handles=hist_ax.get_legend_handles_labels()[0]+dens_ax.get_legend_handles_labels()[0])
                
        plt.savefig(os.path.join(self.save_dir, "{}_{}_density.png".format(model, col_name)))
        plt.close()
    
    def plot_scatter(self, models, cols_names, color):
        assert all([model in ("ae", "clf") for model in models])

        x_data = self.ae_df[cols_names[0]] if models[0] == "ae" else self.clf_df[cols_names[0]]
        y_data = self.ae_df[cols_names[1]] if models[1] == "ae" else self.clf_df[cols_names[1]]

        p = sns.jointplot(x=x_data, y=y_data, kind="reg", color=color, label="Repetetions")
        p.fig.subplots_adjust(top=0.95, left=0.15)
        # plt.scatter(x_data, y_data, color=color, label="Repetetions")
        plt.xlim((min(x_data)-(x_data.mean()-min(x_data))/2, max(x_data)+(max(x_data)-x_data.mean())/2))
        plt.ylim((min(y_data)-(y_data.mean()-min(y_data))/2, max(y_data)+(max(y_data)-y_data.mean())/2))
        plt.suptitle("Scatter Plot")
        plt.xlabel(self.prettify_label(cols_names[0]))
        plt.ylabel(self.prettify_label(cols_names[1]))
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, "{}-{}_{}-{}_scatter.png".format(models[0], cols_names[0], models[1], cols_names[1])))
        plt.close()
        
    def violins(self, cols):
        g1_df = pd.DataFrame({"AUC": self.clf_df[cols[0]]})
        g2_df = pd.DataFrame({"AUC": self.clf_df[cols[1]]})
        g1_df["data_split"] = "train"
        g2_df["data_split"] = "test"
        
        df = pd.concat([g1_df, g2_df], axis=0)
        
        plt.figure(figsize=(7, 8))
        sns.violinplot(data=df, y="AUC", x="data_split")
        plt.title("Violin Plots")
        plt.xlabel(self.prettify_label("data_split"))
        plt.ylabel(self.prettify_label("AUC"))
        plt.savefig(os.path.join(self.save_dir, "{}-{}_violin.png".format(cols[0], cols[1])))
        plt.close()

    def add_autors_results(self, col_name):
        if "test" in col_name.lower():
            for metric in self.metrics:
                if metric.lower() in col_name.lower():
                    plt.plot(self.metrics[metric], 0, 'go', color="red", label="Autors", markersize=8)
                    return True
        return False

    def prettify_label(self, text):
        for word, initial in {"loss": "reconstruction_loss",
                              "auc": "calssification_auc",
                              "accuracy": "calssification_auc",
                              "f1": "calssification_f1"}.items():
            text = text.replace(word, initial)
        return text.title().replace("_", " ").replace("Auc", "AUC")

def main():
    reported_results = {"Cirrhosis": {"AUC": 0.940, "Accuracy": 0.864}, 
                        "Colorectal": {"AUC": 0.803, "Accuracy": 0.728}, 
                        "IBD": {"AUC": 0.955, "Accuracy": 0.773}, 
                        "Obesity": {"AUC": 0.659, "Accuracy": 0.635}, 
                        "T2D": {"AUC": 0.763, "Accuracy": 0.710}, 
                        "WT2D": {"AUC": 0.899, "Accuracy": 0.800}}

    for dataset, metrics in reported_results.items():

        plotter = PlotterClass("results/ae_results_{}".format(dataset), "results/clf_results_{}".format(dataset), dataset, metrics, save_dir="results/results_summary/{}".format(dataset))
        for split, color in zip(["train", "val", "test"], ["darkblue", "olive", "orange"]):
            plotter.plot_density("ae", "{}_loss".format(split), color)
            if split != "val":
                for clf_metric in ["auc", "accuracy", "f1"]:
                    plotter.plot_density("clf", "{}_{}".format(split, clf_metric), color)

        for split, color in zip(["train", "test"], ["darkblue", "orange"]):
            plotter.plot_scatter(["ae", "clf"], ["{}_loss".format(split), "{}_auc".format(split)], color)
        plotter.plot_scatter(["ae", "ae"], ["train_loss", "test_loss"], "purple")
        plotter.plot_scatter(["clf", "clf"], ["train_auc", "test_auc"], "purple")

        plotter.violins(["train_auc", "test_auc"])

if __name__ == "__main__":
    main()