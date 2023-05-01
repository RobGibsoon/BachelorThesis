import argparse
import csv

import pandas as pd

import numpy as np
from utils import append_hyperparams_file, save_preds, append_accuracies_file
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch_geometric.datasets import TUDataset
from utils import NP_SEED, log

DIR = "references"


class ReferenceClassifier:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data = TUDataset(root=f'/tmp/{self.dataset_name}', name=f'{self.dataset_name}')
        filter_split = self.get_csv_idx_split(self.dataset_name, "filter")
        self.X = [self.data[idx] for idx in filter_split]
        self.y = np.array([self.X[i].y.item() for i in range(len(self.X))])
        train_split = self.get_csv_idx_split(self.dataset_name, "train")
        test_split = self.get_csv_idx_split(self.dataset_name, "test")
        train_graphs = [self.X[idx] for idx in train_split]
        test_graphs = [self.X[idx] for idx in test_split]
        self.kernelized_data_training = create_custom_metric(train_graphs, train_graphs)
        self.kernelized_data_test = create_custom_metric(test_graphs, train_graphs)
        """
        self.y = labels of data
        self.X = graphs in data
        training_graphs, test_graphs = graphs in X_train, X_test
        self.X_train = create_custom_metric(training_graphs)
        self.X_test = create_custom_metric(testing_graphs)
        """

        self.X = np.random.rand(12,12)
        self.y = np.ones((12,1))
        self.y[3]-=1
        self.y[7]-=1
        train_graphs, testing_graphs, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                    random_state=NP_SEED)
        self.kernelized_data_training = create_custom_metric(train_graphs, train_graphs)
        self.kernelized_data_test = create_custom_metric(testing_graphs, train_graphs)


    def get_csv_idx_split(self, dn, idx_type):
        file = open(f"../log/index_splits/{dn}_{idx_type}_split.csv", "r")
        idx_split = list(csv.reader(file, delimiter=','))
        parsed_idx_split = [int(elt) for elt in idx_split[0]]
        return parsed_idx_split


    def predict_knn(self):
        """train and predict with knn"""
        k_range = list(range(1, 3))#todo put back to 31
        param_grid = {'metric': ['euclidean', 'manhattan', 'cosine'],
                      'algorithm': ['brute'],
                      'n_neighbors': k_range}
        clf_knn = KNeighborsClassifier(metric='precomputed')

        # perform hyper parameter selection todo: cv=10
        grid_search = GridSearchCV(clf_knn, param_grid, cv=2, scoring='accuracy', return_train_score=False, verbose=1)
        grid_search.fit(self.kernelized_data_training, np.ravel(self.y_train))
        append_hyperparams_file(False, grid_search, clf_knn, self.dataset_name, DIR, ref=True)

        # construct, train optimal model and perform predictions
        knn = KNeighborsClassifier(algorithm=grid_search.best_params_['algorithm'],
                                   metric=grid_search.best_params_['metric'],
                                   n_neighbors=grid_search.best_params_['n_neighbors'])

        knn.fit(self.kernelized_data_training, np.ravel(self.y_train))
        predictions = knn.predict(self.kernelized_data_test)
        test_accuracy = accuracy_score(self.y_test, predictions) * 100
        save_preds(predictions, self.y_test, type(clf_knn).__name__, self.dataset_name, False, ref=True)
        return test_accuracy

    def predict_svm(self):
        """train and predict with svm"""
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                      'gamma': [0.001, 0.01, 0.1, 1, 10, 100]} # maybe kernel=custom
        clf_svm = svm.SVC(kernel='precomputed')

        # perform hyper parameter selection todo: cv=10
        grid_search = GridSearchCV(clf_svm, param_grid, cv=2, scoring='accuracy', error_score='raise',
                                   return_train_score=False, verbose=1)
        grid_search.fit(self.kernelized_data_training, np.ravel(self.y_train))
        append_hyperparams_file(False, grid_search, clf_svm, self.dataset_name, DIR, ref=True)

        # construct, train optimal model and perform predictions
        clf_svm = SVC(C=grid_search.best_params_['C'],
                      gamma=grid_search.best_params_['gamma'])

        clf_svm.fit(self.kernelized_data_training, np.ravel(self.y_train))
        predictions = clf_svm.predict(self.kernelized_data_test)
        test_accuracy = accuracy_score(self.y_test, predictions) * 100
        save_preds(predictions, self.y_test, type(clf_svm).__name__, self.dataset_name, False, ref=True)

        return test_accuracy

    def predict_ann(self):
        return 100.0, "test", "test"


def create_custom_metric(test, train):
    return np.random.rand(len(train), len(test))


if __name__ == "__main__":
    # use this for developing
    dataset_name = "PTC_MR"

    reference_classifier = ReferenceClassifier(dataset_name)
    svm_acc = reference_classifier.predict_svm()
    knn_acc = reference_classifier.predict_knn()
    print(svm_acc)
    print(knn_acc)

    append_accuracies_file(dataset_name, "knn", False, knn_acc, DIR, ref=True)
    append_accuracies_file(dataset_name, "svm", False, svm_acc, DIR, ref=True)

