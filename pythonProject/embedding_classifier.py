import copy
from itertools import chain, combinations

import pandas as pd
import seaborn as sns

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, make_scorer, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils import feature_names, best_subset_cv, get_feature_names, feature_selected_sets

matplotlib.use('TkAgg')


class EmbeddingClassifier:
    def __init__(self, dataset_name):
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
                                                                                random_state=0)

    def predict_knn(self):
        """train and predict with knn"""
        k_range = list(range(1, 31))
        param_grid = {'metric': ['euclidean', 'manhattan', 'cosine'],
                      'algorithm': ['brute'],
                      'n_neighbors': k_range}
        clf_knn = KNeighborsClassifier()

        # perform feature selection
        '''best_subset, best_score = best_subset_cv(clf_knn, self.X_train, self.y_train)
        features = get_feature_names(best_subset)
        print(f"The optimal features selected for KNN were: {features}")
        knn_X_train = self.X_train[:, best_subset]
        knn_X_test = self.X_test[:, best_subset]
        assert (self.X_train.shape[0], len(best_subset)) == knn_X_train.shape'''
        knn_X_train, knn_X_test = feature_selected_sets(clf_knn, self.X_train, self.X_test, self.y_train)

        # perform hyper parameter selection
        clf_knn.fit(knn_X_train, self.y_train)
        grid_search = GridSearchCV(clf_knn, param_grid, cv=10, scoring='accuracy', return_train_score=False, verbose=1)
        grid_search.fit(knn_X_train, self.y_train)
        print(f"The optimal hyper parameters selected for {type(clf_knn).__name__} were: {grid_search.best_params_}")

        # construct, train optimal model and perform predictions
        knn = KNeighborsClassifier(algorithm=grid_search.best_params_['algorithm'],
                                   metric=grid_search.best_params_['metric'],
                                   n_neighbors=grid_search.best_params_['n_neighbors'])

        knn.fit(knn_X_train, self.y_train)
        y_test_hat = knn.predict(knn_X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_hat) * 100

        return test_accuracy

    def predict_svm(self):
        """train and predict with svm"""
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                      'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
                      'kernel': ['rbf', 'sigmoid']}  # ,'linear', 'poly'
        clf_svm = svm.SVC()

        # perform feature selection
        clf_X_train, clf_X_test = feature_selected_sets(clf_svm, self.X_train, self.X_test, self.y_train)

        # perform hyper parameter selection
        grid_search = GridSearchCV(clf_svm, param_grid, cv=10, scoring='accuracy', error_score='raise',
                                   return_train_score=False, verbose=3)
        grid_search.fit(clf_X_train, self.y_train)
        print(f"The optimal hyper parameters selected for {type(clf_svm).__name__} were: {grid_search.best_params_}")

        # construct, train optimal model and perform predictions
        svc = SVC(C=grid_search.best_params_['C'],
                  gamma=grid_search.best_params_['gamma'],
                  kernel=grid_search.best_params_['kernel'])

        svc.fit(clf_X_train, self.y_train)
        y_test_hat = svc.predict(clf_X_test)
        test_accuracy = accuracy_score(self.y_test, y_test_hat) * 100

        return test_accuracy

    def predict_ann(self):
        """train and predict with ann"""

        clf = Network()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(clf.parameters(), lr=0.1)
        epochs = 300
        batch_size = 1
        # todo perform feature selection
        clf_X_train, clf_X_test = feature_selected_sets(clf, self.X_train, self.X_test, self.y_train)

        train_data = Data(clf_X_train, self.y_train)
        test_data = Data(clf_X_test, self.y_test)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = clf(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                print(f'[{epoch + 1}/{epochs}] loss: {running_loss / 2000:.5f}')

        correct, total = 0, 0
        # no need to calculate gradients during inference
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                # calculate output by running through the network
                outputs = clf(inputs)
                # get the predictions
                __, predicted = torch.max(outputs.data, 1)
                # update results
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct // total


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        input_dim = 12
        hidden_layers = 7  # the mean between input_dim + output_dim
        output_dim = 2
        self.linear1 = nn.Linear(input_dim, hidden_layers)
        self.linear2 = nn.Linear(hidden_layers, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x


class Data(Dataset):
    def __init__(self, X_train, y_train):
        self.X = torch.from_numpy(X_train.astype(np.float32))
        self.y = torch.from_numpy(y_train).type(torch.LongTensor)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


if __name__ == "__main__":
    test_accuracies = {}
    dataset_name = 'PTC_MR'
    embedding_classifier = EmbeddingClassifier(dataset_name)
    # test_accuracies['knn'] = embedding_classifier.predict_knn()
    # test_accuracies['svm'] = embedding_classifier.predict_svm()
    test_accuracies['ann'] = embedding_classifier.predict_ann()
    # print(f"Accuracy for our testing dataset with tuning using the SVM model is: {test_accuracies['svm']}")
    # print(f"Accuracy for our testing dataset with tuning using the KNN model is: {test_accuracies['knn']}")
    print(f"Accuracy for our testing dataset with tuning using the ANN model is: {test_accuracies['ann']}")
