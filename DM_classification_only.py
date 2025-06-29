# importing numpy, pandas, and matplotlib
import random
import numpy as np
import pandas as pd
import matplotlib
import tensorflow as tf
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

# importing sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn import cluster
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# importing keras
import keras
import keras.backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.models import Model, load_model

# importing util libraries
import datetime
import time
import math
import os
import importlib

# importing custom library
import DNN_models
import exception_handle

import adaptation_utils

import re
from DM import DeepMicrobiome


if __name__ == '__main__':
    # argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()

    # load data
    load_data = parser.add_argument_group('Loading data')
    load_data.add_argument("-d", "--data", help="prefix of dataset to open (e.g. abundance_Cirrhosis)", type=str,
                        choices=["abundance_Cirrhosis", "abundance_Colorectal", "abundance_IBD",
                                 "abundance_Obesity", "abundance_T2D", "abundance_WT2D",
                                 "marker_Cirrhosis", "marker_Colorectal", "marker_IBD",
                                 "marker_Obesity", "marker_T2D", "marker_WT2D",
                                 ])
    load_data.add_argument("-cd", "--custom_data", help="filename for custom input data under the 'data' folder", type=str,)
    load_data.add_argument("-cl", "--custom_data_labels", help="filename for custom input labels under the 'data' folder", type=str,)
    load_data.add_argument("-p", "--data_dir", help="custom path for both '/data' and '/results' folders", default="")
    load_data.add_argument("-dt", "--dataType", help="Specify data type for numerical values (float16, float32, float64)",
                        default="float64", type=str, choices=["float16", "float32", "float64"])
    dtypeDict = {"float16": np.float16, "float32": np.float32, "float64": np.float64}

    # experiment design
    exp_design = parser.add_argument_group('Experiment design')
    exp_design.add_argument("-s", "--seed", help="random seed for train and test split", type=int, default=0)
    exp_design.add_argument("-r", "--repeat", help="repeat experiment x times by changing random seed for splitting data",
                        default=5, type=int)

    # classification
    classification = parser.add_argument_group('Classification')
    classification.add_argument("-f", "--numFolds", help="The number of folds for cross-validation in the tranining set",
                        default=5, type=int)
    classification.add_argument("-m", "--method", help="classifier(s) to use", type=str, default="all",
                        choices=["all", "svm", "rf", "mlp", "svm_rf"])
    classification.add_argument("-sc", "--svm_cache", help="cache size for svm run", type=int, default=1000)
    classification.add_argument("-t", "--numJobs",
                        help="The number of jobs used in parallel GridSearch. (-1: utilize all possible cores; -2: utilize all possible cores except one.)",
                        default=-2, type=int)
    parser.add_argument("--scoring", help="Metrics used to optimize method", type=str, default='roc_auc',
                        choices=['roc_auc', 'accuracy', 'f1', 'recall', 'precision'])

    # representation learning & dimensionality reduction algorithms
    rl = parser.add_argument_group('Representation learning')
    rl.add_argument("--pca", help="run PCA", action='store_true')
    rl.add_argument("--rp", help="run Random Projection", action='store_true')
    rl.add_argument("--ae", help="run Autoencoder or Deep Autoencoder", action='store_true')
    rl.add_argument("--vae", help="run Variational Autoencoder", action='store_true')
    rl.add_argument("--cae", help="run Convolutional Autoencoder", action='store_true')
    rl.add_argument("--save_rep", help="write the learned representation of the training set as a file", action='store_true')

    # detailed options for representation learning
    ## common options
    common = parser.add_argument_group('Common options for representation learning (SAE,DAE,VAE,CAE)')
    common.add_argument("--aeloss", help="set autoencoder reconstruction loss function", type=str,
                        choices=['mse', 'binary_crossentropy'], default='mse')
    common.add_argument("--ae_oact", help="output layer sigmoid activation function on/off", action='store_true')
    common.add_argument("-a", "--act", help="activation function for hidden layers", type=str, default='relu',
                        choices=['relu', 'sigmoid'])
    common.add_argument("-dm", "--dims",
                        help="Comma-separated dimensions for deep representation learning e.g. (-dm 50,30,20)",
                        type=str, default='50')
    common.add_argument("-e", "--max_epochs", help="Maximum epochs when training autoencoder", type=int, default=2000)
    common.add_argument("-pt", "--patience",
                        help="The number of epochs which can be executed without the improvement in validation loss, right after the last improvement.",
                        type=int, default=20)

    ## AE & DAE only
    AE = parser.add_argument_group('SAE & DAE-specific arguments')
    AE.add_argument("--ae_lact", help="latent layer activation function on/off", action='store_true')

    ## VAE only
    VAE = parser.add_argument_group('VAE-specific arguments')
    VAE.add_argument("--vae_beta", help="weight of KL term", type=float, default=1.0)
    VAE.add_argument("--vae_warmup", help="turn on warm up", action='store_true')
    VAE.add_argument("--vae_warmup_rate", help="warm-up rate which will be multiplied by current epoch to calculate current beta", default=0.01, type=float)

    ## CAE only
    CAE = parser.add_argument_group('CAE-specific arguments')
    CAE.add_argument("--rf_rate", help="What percentage of input size will be the receptive field (kernel) size? [0,1]", type=float, default=0.1)
    CAE.add_argument("--st_rate", help="What percentage of receptive field (kernel) size will be the stride size? [0,1]", type=float, default=0.25)

    # other options
    others = parser.add_argument_group('other optional arguments')
    others.add_argument("--no_trn", help="stop before learning representation to see specified autoencoder structure", action='store_true')
    others.add_argument("--no_clf", help="skip classification tasks", action='store_true')

    
    others.add_argument("--save_loss", help="save the training loss progress", action='store_true')
    
    others.add_argument("--log_per_epoch", help="log representations per epoch", action='store_true')    

    others.add_argument("--subsample", help="log representations per epoch", action='store_true')
    others.add_argument("--subsample_factor", help="number between 0-1 for subsampling the train_val split", type=float, default=1.)
    
    others.add_argument("--name", help="experiment name", type=str)
    
    mlp_fix = parser.add_argument_group('MLP fixing arguments')
    mlp_fix.add_argument("--encoded_file_path", help="the path from which seed, epoch and subsample factor will be derived", type=str)


    args = parser.parse_args()

    print(args)

    # set labels for diseases and controls
    label_dict = {
        # Controls
        'n': 0,
        # Chirrhosis
        'cirrhosis': 1,
        # Colorectal Cancer
        'cancer': 1, 'small_adenoma': 0,
        # IBD
        'ibd_ulcerative_colitis': 1, 'ibd_crohn_disease': 1,
        # T2D and WT2D
        't2d': 1,
        # Obesity
        'leaness': 0, 'obesity': 1,
    }

    # hyper-parameter grids for classifiers
    rf_hyper_parameters = [{'n_estimators': [s for s in range(100, 1001, 200)],
                            'max_features': ['sqrt', 'log2'],
                            'min_samples_leaf': [1, 2, 3, 4, 5],
                            'criterion': ['gini', 'entropy']
                            }, ]
    #svm_hyper_parameters_pasolli = [{'C': [2 ** s for s in range(-5, 16, 2)], 'kernel': ['linear']},
    #                        {'C': [2 ** s for s in range(-5, 16, 2)], 'gamma': [2 ** s for s in range(3, -15, -2)],
    #                         'kernel': ['rbf']}]
    svm_hyper_parameters = [{'C': [2 ** s for s in range(-5, 6, 2)], 'kernel': ['linear']},
                            {'C': [2 ** s for s in range(-5, 6, 2)], 'gamma': [2 ** s for s in range(3, -15, -2)],'kernel': ['rbf']}]
    mlp_hyper_parameters = [{'numHiddenLayers': [1, 2, 3],
                             'epochs': [30, 50, 100, 200, 300],
                             'numUnits': [10, 30, 50, 100],
                             'dropout_rate': [0.1, 0.3],
                             },]
    
    
    train_file_path = os.path.join(os.path.dirname(args.encoded_file_path), os.path.basename(args.encoded_file_path).replace("test", "train"))
    epoch = int(re.search("epoch_(\d+)_", os.path.basename(args.encoded_file_path)).group(1))
    seed = int(re.search("^seed_(\d+)$", os.path.basename(os.path.dirname(args.encoded_file_path))).group(1))
    test_file_path = os.path.join(os.path.dirname(args.encoded_file_path), os.path.basename(args.encoded_file_path).replace("train", "test"))

    experiment_name = "clf_results-exp_name_{}.csv".format(args.name)
    
    result_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(train_file_path))), experiment_name)
    
    logger = adaptation_utils.OnlyCLFLogger(result_file_path, seed=seed, epoch=epoch)

    # run exp function
    def run_exp(seed, global_seed):

        # logger.update_seed(seed=seed)
        
        #fix np.random.seed for reproducibility in numpy processing
        np.random.seed(seed)
        tf.set_random_seed(seed)

        # create an object and load data
        ## no argument founded
        if args.data == None and args.custom_data == None:
            print("[Error] Please specify an input file. (use -h option for help)")
            exit()
        ## provided data
        elif args.data != None:
            dm = DeepMicrobiome(experiment_name=args.name, data=args.data + '.txt', seed=seed, global_seed=global_seed, data_dir=args.data_dir, logger=logger)

            ## specify feature string
            feature_string = ''
            data_string = str(args.data)
            if data_string.split('_')[0] == 'abundance':
                feature_string = "k__"
            if data_string.split('_')[0] == 'marker':
                feature_string = "gi|"

            ## load data into the object
            dm.loadData(feature_string=feature_string, label_string='disease', label_dict=label_dict,
                        dtype=dtypeDict[args.dataType])

        ## user data
        elif args.custom_data != None:

            ### without labels - only conducting representation learning
            if args.custom_data_labels == None:
                dm = DeepMicrobiome(experiment_name=args.name, data=args.custom_data, seed=seed, global_seed=global_seed, data_dir=args.data_dir, logger=logger)
                dm.loadCustomData(dtype=dtypeDict[args.dataType])

            ### with labels - conducting representation learning + classification
            else:
                dm = DeepMicrobiome(experiment_name=args.name, data=args.custom_data, seed=seed, global_seed=global_seed, data_dir=args.data_dir, logger=logger)
                dm.loadCustomDataWithLabels(label_data=args.custom_data_labels, dtype=dtypeDict[args.dataType])

        else:
            exit()

        # if args.subsample and args.subsample_factor < 1 and args.subsample_factor > 0:
        #     dm.subsample_train_val_split(subsample_factor=args.subsample_factor)

        numRLrequired = args.pca + args.ae + args.rp + args.vae + args.cae

        if numRLrequired > 1:
            raise ValueError('No multiple dimensionality Reduction')

        # time check after data has been loaded
        dm.t_start = time.time()

        if not (args.no_trn and args.method != "mlp"):
            # Representation learning (Dimensionality reduction)
            if args.pca:
                dm.pca()
            if args.ae:
                dm.ae(dims=[int(i) for i in args.dims.split(',')], act=args.act, epochs=args.max_epochs, loss=args.aeloss,
                    latent_act=args.ae_lact, output_act=args.ae_oact, patience=args.patience, no_trn=args.no_trn)
            if args.vae:
                dm.vae(dims=[int(i) for i in args.dims.split(',')], act=args.act, epochs=args.max_epochs, loss=args.aeloss, output_act=args.ae_oact,
                    patience= 25 if args.patience==20 else args.patience, beta=args.vae_beta, warmup=args.vae_warmup, warmup_rate=args.vae_warmup_rate, no_trn=args.no_trn)
            if args.cae:
                dm.cae(dims=[int(i) for i in args.dims.split(',')], act=args.act, epochs=args.max_epochs, loss=args.aeloss, output_act=args.ae_oact,
                    patience=args.patience, rf_rate = args.rf_rate, st_rate = args.st_rate, no_trn=args.no_trn)
            if args.rp:
                dm.rp()

        # write the learned representation of the training set as a file
        if args.save_rep and not args.no_trn:
            if numRLrequired == 1:
                rep_path = "results/{}/representations".format(args.data.split("_")[0])
                data_df = pd.DataFrame(dm.X_train)
                data_df["label"] = dm.y_train
                data_df.to_csv(os.path.join(rep_path, "{}_seed_{}_{}_encoded.csv".format(args.name, seed, "train")), index=False)
                
                data_df = pd.DataFrame(dm.X_test)
                data_df["label"] = dm.y_test
                data_df.to_csv(os.path.join(rep_path, "{}_seed_{}_{}_encoded.csv".format(args.name, seed, "test")), index=False)
            
                print("The learned representation of the training set has been saved in '{}'".format(rep_path))
            
                # rep_file = "results/{}/representations".format(args.data.split("_")[0]) + dm.prefix + dm.data + ".csv"
                # train_df = pd.DataFrame(dm.X_train)
                # train_df["label"] = dm.y_train
                # pd.DataFrame(train_df).to_csv(rep_file, index=None)
                # print("The learned representation of the training set has been saved in '{}'".format(rep_file))
            else:
                print("Warning: Command option '--save_rep' is not applied as no representation learning or dimensionality reduction has been conducted.")

        dm.X_train = pd.read_csv(train_file_path).drop(columns="label").values
        dm.y_train = pd.read_csv(train_file_path, usecols=["label"]).values
        dm.X_test = pd.read_csv(test_file_path).drop(columns="label").values
        dm.y_test = pd.read_csv(test_file_path, usecols=["label"]).values
        
        # Classification
        if args.no_clf or (args.data == None and args.custom_data_labels == None):
            print("Classification task has been skipped.")
        else:
            # turn off GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            importlib.reload(keras)

            try:
                # training classification models
                if args.method == "svm":
                    dm.classification(hyper_parameters=svm_hyper_parameters, method='svm', cv=args.numFolds,
                                    n_jobs=args.numJobs, scoring=args.scoring, cache_size=args.svm_cache)
                elif args.method == "rf":
                    dm.classification(hyper_parameters=rf_hyper_parameters, method='rf', cv=args.numFolds,
                                    n_jobs=args.numJobs, scoring=args.scoring)
                elif args.method == "mlp":
                    dm.classification(hyper_parameters=mlp_hyper_parameters, method='mlp', cv=args.numFolds,
                                    n_jobs=args.numJobs, scoring=args.scoring)
                elif args.method == "svm_rf":
                    dm.classification(hyper_parameters=svm_hyper_parameters, method='svm', cv=args.numFolds,
                                    n_jobs=args.numJobs, scoring=args.scoring, cache_size=args.svm_cache)
                    dm.classification(hyper_parameters=rf_hyper_parameters, method='rf', cv=args.numFolds,
                                    n_jobs=args.numJobs, scoring=args.scoring)
                else:
                    dm.classification(hyper_parameters=svm_hyper_parameters, method='svm', cv=args.numFolds,
                                    n_jobs=args.numJobs, scoring=args.scoring, cache_size=args.svm_cache)
                    dm.classification(hyper_parameters=rf_hyper_parameters, method='rf', cv=args.numFolds,
                                    n_jobs=args.numJobs, scoring=args.scoring)
                    dm.classification(hyper_parameters=mlp_hyper_parameters, method='mlp', cv=args.numFolds,
                                    n_jobs=args.numJobs, scoring=args.scoring)
            except ValueError as e:
                if not "Only one class present in y_true." in str(e):
                    raise e
                else:
                    print("Only one class present in the subsampled dataset,\nSkipping the classificaton task for this repetition.")

    if not logger.experiment_already_logged():
        # run experiments
        try:
            run_exp(seed, global_seed=args.seed)        
            K.clear_session()
        except OSError as error:
            exception_handle.log_exception(error)
