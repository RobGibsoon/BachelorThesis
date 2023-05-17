import argparse
import csv

import numpy as np
import pandas as pd
import torch
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader

from ann import mean_score_ann, ANN, Data, train_ann
from references import ReferenceClassifier
from utils import NP_SEED, get_feature_names, all_subsets, log, append_accuracies_file, append_features_file, \
    save_preds, append_hyperparams_file, inputs

np.random.seed(NP_SEED)
DIR = "embedding_classifier"


class EmbeddingClassifier:
    def __init__(self, dataset_name, feature_selection):
        self.dataset_name = dataset_name
        self.feature_selection = feature_selection
        self.data = pd.read_csv(f'embedded_{dataset_name}.csv')
        shape = self.data.shape
        log(f'The dataframe has been read and is of shape {shape[0]}x{shape[1]}', DIR)
        log(f'The dataframe has a total of {self.data.isnull().sum().sum()} NaN values.', DIR)
        log(f'Data read successfully. See head: \n {self.data.head()}', DIR)
        print(f'torch cuda is available: {torch.cuda.is_available()}')
        self.y = self.data['labels'].values
        self.X = self.data.drop('labels', axis=1).values.astype(float)

        # standardizing X
        scaler = StandardScaler()
        scaler.fit(self.X)
        self.X = scaler.transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=NP_SEED)
        save_test_train_split(self.X, self.X_train, self.X_test, dataset_name)
        print('saved test_train splits')

    def predict_knn(self):
        """train and predict with knn"""
        k_range = list(range(1, 31))
        param_grid = {'metric': ['euclidean', 'manhattan', 'cosine'],
                      'algorithm': ['brute'],
                      'n_neighbors': k_range}
        clf_knn = KNeighborsClassifier()

        # perform feature selection
        if self.feature_selection:
            clf_X_train, clf_X_test = feature_selected_sets(clf_knn, self.X_train, self.X_test, self.y_train,
                                                            self.dataset_name)
        else:
            clf_X_train, clf_X_test = self.X_train, self.X_test

        # perform hyper parameter selection
        clf_knn.fit(clf_X_train, self.y_train)
        grid_search = GridSearchCV(clf_knn, param_grid, cv=10, scoring='accuracy', return_train_score=False, verbose=1)
        grid_search.fit(clf_X_train, self.y_train)
        append_hyperparams_file(self.feature_selection, grid_search, clf_knn, self.dataset_name, DIR)

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
            clf_X_train, clf_X_test = feature_selected_sets(clf_svm, self.X_train, self.X_test, self.y_train,
                                                            self.dataset_name)
        else:
            clf_X_train, clf_X_test = self.X_train, self.X_test

        # perform hyper parameter selection
        grid_search = GridSearchCV(clf_svm, param_grid, cv=10, scoring='accuracy', error_score='raise',
                                   return_train_score=False, verbose=1)
        grid_search.fit(clf_X_train, self.y_train)
        append_hyperparams_file(self.feature_selection, grid_search, clf_svm, self.dataset_name, DIR)

        # construct, train optimal model and perform predictions
        clf_svm = SVC(C=grid_search.best_params_['C'],
                      gamma=grid_search.best_params_['gamma'],
                      kernel=grid_search.best_params_['kernel'])

        clf_svm.fit(clf_X_train, self.y_train)
        predictions = clf_svm.predict(clf_X_test)
        test_accuracy = accuracy_score(self.y_test, predictions) * 100
        save_preds(predictions, self.y_test, type(clf_svm).__name__, self.dataset_name, self.feature_selection)

        return test_accuracy

    def predict_ann(self, device):
        """train and predict 5 ANN's"""

        clf = ANN(self.X_train.shape[1])
        if self.feature_selection:
            clf_X_train, clf_X_test = feature_selected_sets(clf, self.X_train, self.X_test, self.y_train,
                                                            self.dataset_name, device)
        else:
            clf_X_train, clf_X_test = self.X_train, self.X_test

        ann_list = [ANN(clf_X_train.shape[1]) for i in range(5)]
        accuracies = np.array([])
        for i, clf_ann in enumerate(ann_list):
            criterion = nn.CrossEntropyLoss()
            epochs = 300
            batch_size = 1

            train_data = Data(clf_X_train, self.y_train)
            test_data = Data(clf_X_test, self.y_test)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_data, batch_size=batch_size)
            accuracy, predictions, labels = train_ann(clf_ann, epochs, criterion, train_loader, test_loader, device)
            accuracies = np.append(accuracies, accuracy)
            save_preds(predictions, self.y_test, type(clf_ann).__name__ + f"{i}", self.dataset_name,
                       self.feature_selection)

            append_accuracies_file(dataset_name, "ann", self.feature_selection, accuracy, DIR, index=i)
        print(accuracies)
        avg_accuracy = np.sum(accuracies) / 5
        high_deviation = np.max(accuracies) - avg_accuracy
        low_deviation = avg_accuracy - min(accuracies)
        return avg_accuracy, high_deviation, low_deviation


def get_best_feature_set(clf, X_train, y, device):
    """returns the best set for classifying using an SVM/KNN clf, uses cross-validation and takes the set with the
    highest mean accuracy"""
    n_features = X_train.shape[1]
    best_score = -np.inf
    best_subset = None
    count = 1
    if isinstance(clf, (KNeighborsClassifier, SVC)):
        for subset in all_subsets(np.arange(n_features)):
            score = cross_val_score(clf, np.reshape(X_train[:, subset], (X_train.shape[0], -1)), y, cv=5).mean()
            if score > best_score:
                best_score, best_subset = score, subset
            log(f'subset {count}/{2 ** n_features - 1}', DIR)
            count += 1
    else:
        for subset in all_subsets(np.arange(n_features)):
            score = mean_score_ann(np.reshape(X_train[:, subset], (X_train.shape[0], -1)), y, device)
            if score > best_score:
                best_score, best_subset = score, subset
            log(f'subset {count}/{2 ** n_features - 1}', DIR)
            count += 1

    log(f"best_subset: {np.array(subset)} with best score: {best_score}", DIR)
    return np.array(subset), best_score


def feature_selected_sets(clf, X_train, X_test, y, dn, device='cpu'):
    """returns the modified training and test sets after performing feature selection on them"""
    best_subset, best_score = get_best_feature_set(clf, X_train, y, device)
    features, count = get_feature_names(best_subset)
    append_features_file(clf, features, count, dn)
    X_train_fs = X_train[:, best_subset]
    X_test_fs = X_test[:, best_subset]
    assert (X_train_fs.shape[0], len(best_subset)) == X_train_fs.shape
    return X_train_fs, X_test_fs


def save_test_train_split(X, X_train, X_test, dataset_name):
    train_indices = np.array([X.tolist().index(X_train[i].tolist()) for i in range(len(X_train))])
    test_indices = np.array([X.tolist().index(X_test[i].tolist()) for i in range(len(X_test))])

    # check that the train_indices and test_indices are correct
    # call numpy.argsort(0) to return the sorted indices of the column 0
    # then use these sorted indices to sort the rows of the same array by column 0
    assert_arr_train = np.array([X[i] for i in train_indices])
    assert_arr_test = np.array([X[i] for i in test_indices])

    sorted_train = assert_arr_train[np.argsort(assert_arr_train[:, 0])]
    sorted_test = assert_arr_test[np.argsort(assert_arr_test[:, 0])]
    assert np.allclose(sorted_train, X_train[np.argsort(X_train[:, 0])])
    assert np.allclose(sorted_test, X_test[np.argsort(X_test[:, 0])])
    with open(f'log/index_splits/{dataset_name}_train_split.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(train_indices)
    file.close()
    with open(f'log/index_splits/{dataset_name}_test_split.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(test_indices)
    file.close()
    log(f"{dataset_name}\n"
        f"he optimal train split: {train_indices}\n"
        f"The optimal test split: {test_indices}\n", DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=str,
                        help='The index of which command should be completed according to the inputs in utils.')
    args = parser.parse_args()
    if args.idx is None:
        raise argparse.ArgumentError(None, "Please a possible index from 0-26.")

    idx = int(args.idx)
    parameters = inputs[idx]
    log(f'{parameters}', DIR)
    dataset_name = parameters[0]
    clf_model = parameters[1]
    is_fs = parameters[2]
    is_reference = parameters[3]

    # device used for ANNs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embedding_classifier = EmbeddingClassifier(dataset_name, feature_selection=is_fs)
    if not is_reference:
        if clf_model.lower() == 'knn':
            acc = embedding_classifier.predict_knn()
            log(f"Accuracy for our testing {dataset_name} dataset with tuning using the KNN model is: {acc}", DIR)
            append_accuracies_file(dataset_name, clf_model, is_fs, acc, DIR)
        elif clf_model.lower() == 'svm':
            acc = embedding_classifier.predict_svm()
            log(f"Accuracy for our testing {dataset_name} dataset with tuning using the SVM model is: {acc}", DIR)
            append_accuracies_file(dataset_name, clf_model, is_fs, acc, DIR)
        elif clf_model.lower() == 'ann':
            print(f'using ann and {dataset_name} and {is_fs}')
            avg_accuracy, high_deviation, low_deviation = embedding_classifier.predict_ann(device)
            log(f"Average accuracy for our testing {dataset_name} dataset with tuning using the ANN model is: {avg_accuracy} "
                f"with highest being +{round(high_deviation, 2)} and the lowest -{round(low_deviation, 2)}", DIR)
            append_accuracies_file(dataset_name, "ann_avg", is_fs, avg_accuracy, DIR)
        else:
            raise argparse.ArgumentTypeError('Invalid classifier. Pick between knn, svm or ann.')

        log(f"Used feature selection: {False if is_fs == False else True}", DIR)
    else:
        ref_dir = "references"
        reference_classifier = ReferenceClassifier(dataset_name)
        if clf_model.lower() == 'knn':
            acc = reference_classifier.predict_knn()
            log(f"Accuracy for our testing {dataset_name} dataset with tuning using the KNN model is: {acc}", DIR)
            append_accuracies_file(dataset_name, clf_model, is_fs, acc, ref_dir, ref=True)
        elif clf_model.lower() == 'svm':
            acc = reference_classifier.predict_svm()
            log(f"Accuracy for our testing {dataset_name} dataset with tuning using the SVM model is: {acc}", DIR)
            append_accuracies_file(dataset_name, clf_model, is_fs, acc, ref_dir, ref=True)
        elif clf_model.lower() == 'ann':
            avg_accuracy, high_deviation, low_deviation = reference_classifier.predict_ann(device)
            log(f"Average accuracy for our testing {dataset_name} dataset with tuning using the ANN model is: {avg_accuracy} "
                f"with highest being +{round(high_deviation, 2)} and the lowest -{round(low_deviation, 2)}", DIR)
            append_accuracies_file(dataset_name, "ann_avg", is_fs, avg_accuracy, ref_dir, ref=True)
        else:
            raise argparse.ArgumentTypeError('Invalid classifier. Pick between knn, svm or ann.')
