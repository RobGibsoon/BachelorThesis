from datetime import datetime
from multiprocessing import Pool
from time import time

import numpy as np
from cyged.graph_pkg_core import GED
from cyged.graph_pkg_core import Graph
from cyged.graph_pkg_core.edit_cost.edit_cost_vector import EditCostVector
from cyged.graph_pkg_core.graph.edge import Edge
from cyged.graph_pkg_core.graph.label.label_edge import LabelEdge
from cyged.graph_pkg_core.graph.label.label_node_vector import LabelNodeVector
from cyged.graph_pkg_core.graph.node import Node
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

from utils import append_hyperparams_file, save_preds, append_accuracies_file, log, get_csv_idx_split

DIR = "references"


class ReferenceClassifier:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data = TUDataset(root=f'/tmp/{self.dataset_name}', name=f'{self.dataset_name}')
        filter_split = get_csv_idx_split(self.dataset_name, "filter")
        self.X = [self.data[idx] for idx in filter_split]
        self.y = np.array([self.X[i].y.item() for i in range(len(self.X))])
        train_split = get_csv_idx_split(self.dataset_name, "train")
        test_split = get_csv_idx_split(self.dataset_name, "test")
        train_graphs = [self.X[idx] for idx in train_split]
        test_graphs = [self.X[idx] for idx in test_split]
        self.y_train = np.array([self.y[idx] for idx in train_split])
        self.y_test = np.array([self.y[idx] for idx in test_split])
        alpha_values = np.arange(0.05, 1.0, 0.1)
        self.kernelized_data_train = [create_custom_metric(train_graphs, train_graphs, alpha) for alpha in
                                      alpha_values]
        log(f'Finished generating train-data kernel for {self.dataset_name}', DIR)
        self.kernelized_data_test = [create_custom_metric(test_graphs, train_graphs, alpha) for alpha in alpha_values]
        log(f'Finished generating test-data kernel for {self.dataset_name}', DIR)

    def predict_knn(self):
        """train and predict with knn"""
        log(f'running references on: ({self.dataset_name}, knn)', DIR)
        k_range = list(range(1, 31))
        param_grid = {'algorithm': ['brute'],
                      'n_neighbors': k_range}
        log('Finding best alpha on ptc_mr', DIR)

        # find best alpha
        clf_knn = KNeighborsClassifier(metric='precomputed')
        best_kernel_index = self.find_best_alpha(clf_knn, self.kernelized_data_train, self.y_train, param_grid)

        # perform gridsearch using kernel with best alpha
        grid_search = GridSearchCV(clf_knn, param_grid, cv=10, scoring='accuracy', return_train_score=False,
                                   verbose=1, n_jobs=-1)
        start_time = time()
        grid_search.fit(self.kernelized_data_train[best_kernel_index], np.ravel(self.y_train))
        grid_search_time = time() - start_time

        # do final fitting with best params
        best_knn = KNeighborsClassifier(algorithm=grid_search.best_params_['algorithm'],
                                        n_neighbors=grid_search.best_params_['n_neighbors'],
                                        metric='precomputed')
        start_time = time()
        best_knn.fit(self.kernelized_data_train[best_kernel_index], np.ravel(self.y_train))
        clf_time = time() - start_time

        # perform prediction and log data
        log(f'finished knn fitting on {self.dataset_name} and found best alpha {np.arange(0.05, 1.0, 0.1)[best_kernel_index]}',
            DIR)
        append_hyperparams_file(False, grid_search, best_knn, self.dataset_name, DIR, ref=True)
        predictions = best_knn.predict(self.kernelized_data_test[best_kernel_index])
        test_accuracy = accuracy_score(self.y_test, predictions) * 100
        save_preds(predictions, self.y_test, type(best_knn).__name__, self.dataset_name, False, ref=True)
        grid_search_time = datetime.utcfromtimestamp(grid_search_time).strftime('%H:%M:%S.%f')[:-4]
        clf_time = datetime.utcfromtimestamp(clf_time).strftime('%H:%M:%S.%f')[:-4]
        log(f"Reference Gridsearch time on {self.dataset_name} svm: {grid_search_time} \n"
            f"Reference Classification time on {self.dataset_name} svm {clf_time}: ", "time")

        return test_accuracy

    def predict_svm(self):
        """train and predict with svm"""
        log(f'running references on: ({self.dataset_name}, knn)', DIR)

        self.kernelized_data_train = [data_training * (-1) for data_training in
                                      self.kernelized_data_train]
        self.kernelized_data_test = [data_test * (-1) for data_test in
                                     self.kernelized_data_test]

        # find best alpha with less extensive param_grid
        clf_svm = svm.SVC(kernel='precomputed')
        small_param_grid = {'C': [0.01, 0.1, 1, 10]}
        best_kernel_index = self.find_best_alpha(clf_svm, self.kernelized_data_train, self.y_train, small_param_grid)

        # do more detailed optimization now that we have the best kernel (alpha)
        big_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        clf_svm = svm.SVC(kernel='precomputed')
        detailed_grid_search = GridSearchCV(clf_svm, big_param_grid, cv=10, scoring='accuracy', error_score='raise',
                                            return_train_score=False, verbose=1, n_jobs=-1)
        log(f'Completing svm girdsearch on {self.dataset_name} with detailed param grid.', DIR)
        start_time = time()
        detailed_grid_search.fit(self.kernelized_data_train[best_kernel_index], np.ravel(self.y_train))
        grid_search_time = time() - start_time
        log(f'finished detailed svm gridsearch on {self.dataset_name}', DIR)
        append_hyperparams_file(False, detailed_grid_search, clf_svm, self.dataset_name, DIR, ref=True)

        # do final fitting with best params
        best_svm = SVC(C=detailed_grid_search.best_params_['C'],
                       kernel='precomputed')
        log(f'fiting best_svm on {self.dataset_name}', DIR)
        start_time = time()
        best_svm.fit(self.kernelized_data_train[best_kernel_index], np.ravel(self.y_train))
        clf_time = time() - start_time

        # perform prediction and log data
        grid_search_time = datetime.utcfromtimestamp(grid_search_time).strftime('%H:%M:%S.%f')[:-4]
        clf_time = datetime.utcfromtimestamp(clf_time).strftime('%H:%M:%S.%f')[:-4]
        log(f"Reference Gridsearch time on {self.dataset_name} svm: {grid_search_time} \n"
            f"Reference Classification time on {self.dataset_name} svm {clf_time}: ", "time")

        predictions = best_svm.predict(self.kernelized_data_test[best_kernel_index])
        test_accuracy = accuracy_score(self.y_test, predictions) * 100
        save_preds(predictions, self.y_test, type(best_svm).__name__, self.dataset_name, False, ref=True)

        return test_accuracy

    def find_best_alpha(self, clf, kernel, y_train, small_param_grid):
        """selects the best alpha from the different provided kernels by using grid search"""
        prev_score = 0
        best_kernel_index = 0
        for i, cur_kernel in enumerate(kernel):
            grid_search = GridSearchCV(clf, small_param_grid, cv=5, scoring='accuracy', error_score='raise',
                                       return_train_score=True, verbose=1, n_jobs=-1)
            print('y_train shape: ', y_train.shape)
            print('cur_kernel shape: ', cur_kernel.shape, '\n')
            grid_search.fit(cur_kernel, np.ravel(y_train))
            scores = grid_search.cv_results_['mean_test_score']
            mean_score = np.mean(scores)

            if mean_score > prev_score:
                # update best score and alpha if better values are found
                prev_score = mean_score
                best_kernel_index = i

        append_hyperparams_file(False, f"Found best alpha: {np.arange(0.05, 1.0, 0.1)[best_kernel_index]}", clf,
                                self.dataset_name, DIR, ref=True)
        log(f'Completed small svm girdsearch on {self.dataset_name} with small param grid and found '
            f'{np.arange(0.05, 1.0, 0.1)[best_kernel_index]}', DIR)
        return best_kernel_index


def create_custom_metric(test, train, alpha):
    rows = len(test)
    cols = len(train)
    res_mat = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            res_mat[i, j] = graph_edit_distance(test[i], train[j], alpha)
    assert np.abs(np.sum(res_mat)) > 0
    print("created metric")
    return res_mat


def worker(args):
    test, train, alpha = args
    return graph_edit_distance(test, train, alpha)


def create_custom_metric_parallel(test, train, alpha):
    rows = len(test)
    cols = len(train)

    # Preparing tuples
    tuples = [(test[i], train[j], alpha) for i in range(rows) for j in range(cols)]

    # Using multiprocessing pool
    with Pool() as pool:
        res = pool.map(worker, tuples)

    # Converting list to array and reshaping to matrix
    res_mat = np.array(res).reshape(rows, cols)
    assert np.abs(np.sum(res_mat)) > 0
    return res_mat


def graph_edit_distance(gr_1, gr_2, alpha):
    ged = GED(EditCostVector(1., 1., 1., 1., "euclidean", alpha=alpha))
    ant_gr_1 = data_to_custom_graph(gr_1)
    ant_gr_2 = data_to_custom_graph(gr_2)
    edit_cost = ged.compute_edit_distance(ant_gr_1, ant_gr_2)
    return edit_cost


def data_to_custom_graph(data: Data):
    n = data.num_nodes
    m = data.num_edges

    graph = Graph("", "", n)

    # Add nodes to the custom graph
    for i, node_feat in enumerate(data.x.numpy().astype(np.double)):
        graph.add_node(Node(i, LabelNodeVector(node_feat)))

    # Add edges to the custom graph
    edge_index = data.edge_index.numpy()
    for i in range(0, edge_index.shape[1], 2):
        src, dest = edge_index[:, i]
        graph.add_edge(Edge(src, dest, LabelEdge(0)))

    return graph


if __name__ == "__main__":
    # use this for developing
    dataset_name = "MUTAG"

    reference_classifier = ReferenceClassifier(dataset_name)
    svm_acc = reference_classifier.predict_svm()
    knn_acc = reference_classifier.predict_knn()
    print(svm_acc)
    print(knn_acc)

    append_accuracies_file(dataset_name, "knn", False, knn_acc, DIR, ref=True)
    append_accuracies_file(dataset_name, "svm", False, svm_acc, DIR, ref=True)
