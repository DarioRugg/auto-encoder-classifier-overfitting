# from visualization.plotting import PlotterClass
from plotting import PlotterClass


def main():
    reported_results = {"Cirrhosis-SAE": {"AUC": 0.928, "Accuracy": 0.821}, 
                        "Cirrhosis-DAE": {"AUC": 0.903, "Accuracy": 0.809}, 
                        "Cirrhosis-VAE": {"AUC": 0.891, "Accuracy": 0.792},
                        "Colorectal-SAE": {"AUC": 0.799, "Accuracy": 0.752}, 
                        "Colorectal-VAE": {"AUC": 0.737, "Accuracy": 0.696}, 
                        "Colorectal-CAE": {"AUC": 0.789, "Accuracy": 0.744}, 
                        "IBD-DAE": {"AUC": 0.911, "Accuracy": 0.855}, 
                        "IBD-VAE": {"AUC": 0.899, "Accuracy": 0.818}, 
                        "IBD-CAE": {"AUC": 0.929, "Accuracy": 0.882},
                        "T2D_SAE": {"AUC": 0.762, "Accuracy": 0.664},   
                        "T2D_DAE": {"AUC": 0.702, "Accuracy": 0.649}, 
                        "T2D_VAE": {"AUC": 0.719, "Accuracy": 0.664},
                        "Obesity-SAE": {"AUC": 0.658, "Accuracy": 0.624}, 
                        "Obesity-VAE": {"AUC": 0.599, "Accuracy": 0.639}, 
                        "Obesity-CAE": {"AUC": 0.622, "Accuracy": 0.655}}

    for dataset, metrics in reported_results.items():
        plotter = PlotterClass(None, "results/clf_results_{}".format(dataset), dataset, metrics, save_dir="results/results_summary/{}".format(dataset))

        split = "test"
        for clf_metric, color in zip(["auc", "accuracy"], ["darkblue", "orange"]):
            plotter.plot_density("clf", "{}_{}".format(split, clf_metric), color)
        # plotter.plot_scatter(["clf", "clf"], ["train_auc", "test_auc"], "purple")

        plotter.violins(["train_auc", "test_auc"])

if __name__ == "__main__":
    main()