import argparse
import copy
import sys
from itertools import chain, combinations

import pandas as pd
import seaborn as sns

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader
from utils import feature_selected_sets, ANN, Data, \
    train_ann, save_preds, NP_SEED

matplotlib.use('TkAgg')

np.random.seed(NP_SEED)


class EmbeddingClassifier:
    def __init__(self, dataset_name, feature_selection):
        self.dataset_name = dataset_name
        self.feature_selection = feature_selection
        self.data = pd.read_csv(f'C:/Users/Robin/BachelorThesis/BachelorThesis/embedded_{dataset_name}.csv')
        shape = self.data.shape
        print(f'The dataframe has been read and is of shape {shape[0]}x{shape[1]}')
        print(f'The dataframe has a total of {self.data.isnull().sum().sum()} NaN values.')
        print(f'Data read successfully. See head: \n {self.data.head()}')

        self.y = self.data['labels'].values
        self.X = self.data.drop('labels', axis=1).values.astype(float)

        plt.subplots(figsize=(12, 5))
        correlation = self.data.corr(numeric_only=True)
        sns.heatmap(correlation, annot=True, cmap='RdPu')
        plt.title('Correlation between the variables')
        plt.xticks(rotation=45)
        # plt.show()

        # standardizing X
        scaler = StandardScaler()
        scaler.fit(self.X)
        self.X = scaler.transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=NP_SEED)

    def predict_knn(self):
        """train and predict with knn"""
        k_range = list(range(1, 31))
        param_grid = {'metric': ['euclidean', 'manhattan', 'cosine'],
                      'algorithm': ['brute'],
                      'n_neighbors': k_range}
        clf_knn = KNeighborsClassifier()

        # perform feature selection
        if self.feature_selection:
            clf_X_train, clf_X_test = feature_selected_sets(clf_knn, self.X_train, self.X_test, self.y_train)
        else:
            clf_X_train, clf_X_test = self.X_train, self.X_test

        # perform hyper parameter selection
        clf_knn.fit(clf_X_train, self.y_train)
        grid_search = GridSearchCV(clf_knn, param_grid, cv=10, scoring='accuracy', return_train_score=False, verbose=1)
        grid_search.fit(clf_X_train, self.y_train)
        print(f"The optimal hyper parameters selected for {type(clf_knn).__name__} were: {grid_search.best_params_}")

        # construct, train optimal model and perform predictions
        knn = KNeighborsClassifier(algorithm=grid_search.best_params_['algorithm'],
                                   metric=grid_search.best_params_['metric'],
                                   n_neighbors=grid_search.best_params_['n_neighbors'])

        knn.fit(clf_X_train, self.y_train)
        predictions = knn.predict(clf_X_test)
        test_accuracy = accuracy_score(self.y_test, predictions) * 100
        save_preds(predictions, self.y_test, type(clf_knn).__name__, self.dataset_name, self.feature_selection)
        return test_accuracy

    def predict_svm(self):
        """train and predict with svm"""
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                      'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
                      'kernel': ['rbf', 'sigmoid']}
        clf_svm = svm.SVC()

        # perform feature selection
        if self.feature_selection:
            clf_X_train, clf_X_test = feature_selected_sets(clf_svm, self.X_train, self.X_test, self.y_train)
        else:
            clf_X_train, clf_X_test = self.X_train, self.X_test

        # perform hyper parameter selection
        grid_search = GridSearchCV(clf_svm, param_grid, cv=10, scoring='accuracy', error_score='raise',
                                   return_train_score=False, verbose=1)
        grid_search.fit(clf_X_train, self.y_train)
        print(f"The optimal hyper parameters selected for {type(clf_svm).__name__} were: {grid_search.best_params_}")

        # construct, train optimal model and perform predictions
        clf_svm = SVC(C=grid_search.best_params_['C'],
                      gamma=grid_search.best_params_['gamma'],
                      kernel=grid_search.best_params_['kernel'])

        clf_svm.fit(clf_X_train, self.y_train)
        predictions = clf_svm.predict(clf_X_test)
        test_accuracy = accuracy_score(self.y_test, predictions) * 100
        save_preds(predictions, self.y_test, type(clf_svm).__name__, self.dataset_name, self.feature_selection)

        return test_accuracy

    def predict_ann(self):
        """train and predict with ann"""

        clf = ANN(self.X_train.shape[1])
        if self.feature_selection:
            clf_X_train, clf_X_test = feature_selected_sets(clf, self.X_train, self.X_test, self.y_train)
        else:
            clf_X_train, clf_X_test = self.X_train, self.X_test
        clf_ann = ANN(clf_X_train.shape[1])

        criterion = nn.CrossEntropyLoss()
        epochs = 300
        batch_size = 1

        train_data = Data(clf_X_train, self.y_train)
        test_data = Data(clf_X_test, self.y_test)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=batch_size)
        accuracy, predictions, labels = train_ann(clf_ann, epochs, criterion, train_loader, test_loader)
        save_preds(predictions, self.y_test, type(clf_ann).__name__, self.dataset_name, self.feature_selection)
        return accuracy


if __name__ == "__main__":
    dataset_name = "PTC_MR"
    embedding_classifier = EmbeddingClassifier(dataset_name, feature_selection=True)
    acc = embedding_classifier.predict_ann()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dn', type=str, help='The name of the dataset to be classified.')
    # parser.add_argument('--clf', type=str, default='knn', help='Which classifier model should be used. Choose '
    #                                                            'between: svm, knn or ann')
    # parser.add_argument('--fs', default=False, help='Whether the feature selection is wished. Default is true')
    #
    # args = parser.parse_args()
    # if args.dn is None:
    #     raise argparse.ArgumentError(None, "Please enter the required arguments: --dn, --clf and optionally --fs")
    #
    # dataset_name = args.dn
    # feature_selection = args.fs
    # clf_model = args.clf
    # embedding_classifier = EmbeddingClassifier(dataset_name, feature_selection=feature_selection)
    #
    # if clf_model.lower() == 'knn':
    #     acc = embedding_classifier.predict_knn()
    #     print(f"Accuracy for our testing {dataset_name} dataset with tuning using the KNN model is: {acc}")
    # elif clf_model.lower() == 'svm':
    #     acc = embedding_classifier.predict_svm()
    #     print(f"Accuracy for our testing {dataset_name} dataset with tuning using the SVM model is: {acc}")
    # elif clf_model.lower() == 'ann':
    #     acc = embedding_classifier.predict_ann()
    #     print(f"Accuracy for our testing {dataset_name} dataset with tuning using the ANN model is: {acc}")
    # else:
    #     raise argparse.ArgumentTypeError('Invalid classifier. Pick between knn, svm or ann.')
    #
    # print(f"Used feature selection: {False if feature_selection == False else True}")
