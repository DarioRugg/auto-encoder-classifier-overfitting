import os
import keras
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import sklearn
import pickle

import tensorflow as tf
from keras.models import Model

# Import writer class from csv module
from csv import writer


class CompleteLogger:
    def __init__(self, ae_file, clf_file, experiment_number, loss_file=None) -> None:
        self.ae_file = ae_file
        self.clf_file = clf_file
        self.loss_path = loss_file

        self.exp_num = experiment_number
        self.seed = None
    
    def update_seed(self, seed):
        self.seed = seed

    def log_ae_metrics(self, model: keras.models.Model, train: np.ndarray, test: np.ndarray, val: np.ndarray):

        if not os.path.isfile(self.ae_file):
            if os.path.split(self.ae_file)[0]:
                os.makedirs(os.path.split(self.ae_file)[0], exist_ok=True)
            with open(self.ae_file, "w") as f:
                writer(f).writerow(['exp_num', "seed", "train_loss", "val_loss", "test_loss"])
        
        print("  ----> results shape", model.predict(test).flatten().shape)
        print("  ----> data shape   ", test.flatten().shape)

        row_to_append = [self.exp_num,
                         self.seed,
                         mean_squared_error(train.flatten(), model.predict(train).flatten()),
                         mean_squared_error(test.flatten(), model.predict(test).flatten()),
                         mean_squared_error(val.flatten(), model.predict(val).flatten())]

        with open(self.ae_file, 'a') as f_object:
            
            writer(f_object).writerow(row_to_append)

    def log_clf_metrics(self, model, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):

        if not os.path.isfile(self.clf_file):
            if os.path.split(self.clf_file)[0]:
                os.makedirs(os.path.split(self.clf_file)[0], exist_ok=True)
            with open(self.clf_file, "w") as f:
                writer(f).writerow(['exp_num', "seed", "train_auc", "train_accuracy", "train_f1", "test_auc", "test_accuracy", "test_f1"])
        
        # Evaluate performance of the best model on test set
        y_train_pred = model.predict(x_train)
        y_train_prob = model.predict_proba(x_train)
        y_test_pred = model.predict(x_test)
        y_test_prob = model.predict_proba(x_test)

        row_to_append = [self.exp_num,
                         self.seed,
                         sklearn.metrics.roc_auc_score(y_train, y_train_prob[:, 1]),
                         sklearn.metrics.accuracy_score(y_train, y_train_pred),
                         sklearn.metrics.f1_score(y_train, y_train_pred),
                         sklearn.metrics.roc_auc_score(y_test, y_test_prob[:, 1]),
                         sklearn.metrics.accuracy_score(y_test, y_test_pred),
                         sklearn.metrics.f1_score(y_test, y_test_pred) ]

        with open(self.clf_file, 'a') as f_object:
    
            # Pass this file object to csv.writer()
            # and get a writer object
            writer(f_object).writerow(row_to_append)
    
    
    def save_loss(self, loss_object):
        path = os.path.join(self.loss_path, "seed_{}.pkl".format(self.seed))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as handle:
            pickle.dump(loss_object, handle, protocol=pickle.HIGHEST_PROTOCOL)

  
class EncodingCallback(tf.keras.callbacks.Callback):
    def __init__(self, encoder_model: keras.models.Model, encoding_dir: str, data_dict: dict):
        super().__init__()
        self.model = encoder_model
        self.encoding_dir = encoding_dir
        self.data_dict = data_dict
        
        os.makedirs(self.encoding_dir, exist_ok=True)
            
    def on_epoch_end(self, epoch, logs=None):
        if epoch%8 == 0:
            layer_idx = int((len(self.model.layers) - 1) / 2)
            encoder = Model(self.model.layers[0].input, self.model.layers[layer_idx].output)
        
            for split in ["train", "test"]:
                data_df = pd.DataFrame(encoder.predict(self.data_dict["x_{}".format(split)]))
                data_df["label"] = self.data_dict["y_{}".format(split)]
                data_df.to_csv(os.path.join(self.encoding_dir, "epoch_{}_{}_encoded.csv".format(epoch, split)), index=False)
                
        return super().on_epoch_end(epoch, logs)
                                    

class OnlyCLFLogger():
    def __init__(self, result_file, epoch, seed) -> None:
        self.file = result_file
        
        self.epoch = epoch
        self.seed = seed
    
    def log_clf_metrics(self, model, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
        if not os.path.isfile(self.file):
            os.makedirs(os.path.dirname(self.file), exist_ok=True)
            with open(self.file, "w") as f:
                writer(f).writerow(["seed", "epoch", "train_auc", "train_accuracy", "train_f1", "test_auc", "test_accuracy", "test_f1"])
        
        # Evaluate performance of the best model on test set
        y_train_pred = model.predict(x_train)
        y_train_prob = model.predict_proba(x_train)
        y_test_pred = model.predict(x_test)
        y_test_prob = model.predict_proba(x_test)

        row_to_append = [self.seed,
                         self.epoch,
                         sklearn.metrics.roc_auc_score(y_train, y_train_prob[:, 1]),
                         sklearn.metrics.accuracy_score(y_train, y_train_pred),
                         sklearn.metrics.f1_score(y_train, y_train_pred),
                         sklearn.metrics.roc_auc_score(y_test, y_test_prob[:, 1]),
                         sklearn.metrics.accuracy_score(y_test, y_test_pred),
                         sklearn.metrics.f1_score(y_test, y_test_pred) ]

        with open(self.file, 'a') as f_object:
    
            # Pass this file object to csv.writer()
            # and get a writer object
            writer(f_object).writerow(row_to_append)
    
    def experiment_already_logged(self):
        if not os.path.isfile(self.file):
            return False
        else:
            loggs_df = pd.read_csv(self.file)
            num_logs = sum((loggs_df["seed"]==self.seed)&(loggs_df["epoch"]==self.epoch))
            assert num_logs <= 1
            return True if num_logs == 1 else False
