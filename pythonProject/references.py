import argparse

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

        """
        self.y = labels of data
        self.X = graphs in data
        training_graphs, test_graphs = graphs in X_train, X_test
        self.X_train = create_custom_metric(training_graphs)
        self.X_test = create_custom_metric(testing_graphs)
        """

        # TODO: Filter the datset and set the train/test splits based on dataset_name using the saved splits

        self.X = np.random.rand(12,12)
        self.y = np.ones((12,1))
        self.y[3]-=1
        self.y[7]-=1
        training_graphs, testing_graphs, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                    random_state=NP_SEED)
        self.kernelized_data_training = create_custom_metric(training_graphs, training_graphs)
        self.kernelized_data_test = create_custom_metric(testing_graphs, training_graphs)


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


def create_custom_metric(train, test):
    n = train.shape[0]
    return np.random.rand(len(train), len(test))


if __name__ == "__main__":
    """parser = argparse.ArgumentParser()
    parser.add_argument('--dn', type=str, help='The name of the dataset to be classified.')
    parser.add_argument('--clf', type=str, default='knn', help='Which classifier model should be used. Choose '
                                                               'between: svm, knn or ann')
    args = parser.parse_args()
    if args.dn is None:
        raise argparse.ArgumentError(None, "Please enter the required arguments: --dn and --clf ")
    dataset_name = args.dn
    clf_model = args.clf"""

    dataset_name = "PTC_MR"

    reference_classifier = ReferenceClassifier(dataset_name)
    svm_acc = reference_classifier.predict_svm()
    knn_acc = reference_classifier.predict_knn()
    print(svm_acc)
    print(knn_acc)

    append_accuracies_file(dataset_name, "knn", False, knn_acc, DIR, ref=True)
    append_accuracies_file(dataset_name, "svm", False, svm_acc, DIR, ref=True)

