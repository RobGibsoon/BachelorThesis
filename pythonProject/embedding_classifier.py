import copy

import pandas as pd

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, make_scorer, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

dataset_name = 'PTC_MR'


class EmbeddingClassifier:
    def __init__(self):
        self.data = pd.read_csv(f'C:/Users/Robin/BachelorThesis/BachelorThesis/embedded_{dataset_name}.csv').to_numpy()
        self.y = self.data['labels'].values
        self.X = self.data.drop('labels', axis=1).values
        std = np.std(self.X, axis=0)
        mean = np.mean(self.X, axis=0)
        self.X = (self.X -  mean)/std

    def predict_knn(self):
        """train and predict with knn"""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=0)
        clf_knn = KNeighborsClassifier(n_neighbors=2)
        clf_knn.fit(X_train, y_train)
        print(f'accuracy score: {accuracy_score(clf_knn.predict(X_test), y_test)}')

    def predict_svm(self):
        """train and predict with svm"""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)

        # Set up the parameter grid for the grid search
        # param_grid = {'C': [0.1, 1, 10],
        #               'gamma': [0.1, 1, 10],
        #               'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
        param_grid = {'C': [0.1, 1, 10],
                      'gamma': [0.1, 1, 10],
                      'kernel': ['poly']}

        print('Set up the grid search')
        clf_svm = svm.SVC()
        grid_search = GridSearchCV(clf_svm, param_grid=param_grid, cv=5, scoring='accuracy', verbose=3)

        print('Fit the grid search on the training set')
        grid_search.fit(X_train, y_train)

        print('Get the best model from the grid search and train it on the training set using 5-fold cross validation')
        print(f'Best hyper-parameters: {grid_search.best_params_}\n')
        best_model = grid_search.best_estimator_
        #
        # kf = KFold(n_splits=5, shuffle=True, random_state=1111)
        # splits = kf.split(X_train)

        # current_accuracy = 0
        # split_count = 0
        # for train_index, val_index in splits:
        #     print(f'Training split {split_count}')
        #     split_count += 1
        #     X_train_val, y_train_val = X_train[train_index], y_train[train_index]
        #     X_val, y_val = X_train[val_index], y_train[val_index]
        #
        #     # Fit the random forest model
        #     gs_model.fit(X_train_val, y_train_val)
        #
        #     # Make predictions, and print the accuracy
        #     predictions = gs_model.predict(X_val)
        #     print("Split accuracy: " + str(log_loss(y_val, predictions)))
        #     if log_loss(y_val, predictions) > current_accuracy:
        #         best_model = copy.copy(gs_model)

        print('Evaluate the best model on the test set')
        y_pred_test = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        print(f'Final accuracy: {test_accuracy}')


if __name__ == "__main__":
    embedding_classifier = EmbeddingClassifier()
    embedding_classifier.predict_svm()
