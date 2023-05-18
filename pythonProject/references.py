import csv

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

from utils import append_hyperparams_file, save_preds, append_accuracies_file, log

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
        self.y_train = [self.y[idx] for idx in train_split]
        self.y_test = [self.y[idx] for idx in test_split]
        alpha_values = np.arange(0.05, 1.0, 0.1)
        self.kernelized_data_training = [create_custom_metric(train_graphs, train_graphs, alpha) for alpha in
                                         alpha_values]
        self.kernelized_data_test = [create_custom_metric(test_graphs, train_graphs, alpha) for alpha in alpha_values]

    def get_csv_idx_split(self, dn, idx_type):
        file = open(f"log/index_splits/{dn}_{idx_type}_split.csv", "r")
        idx_split = list(csv.reader(file, delimiter=','))
        parsed_idx_split = [int(elt) for elt in idx_split[0]]
        return parsed_idx_split

    def predict_knn(self):
        """train and predict with knn"""
        best_kernel_index = 0
        prev_score = 0
        best_knn = None
        best_grid_search = None

        k_range = list(range(1, 31))
        param_grid = {'algorithm': ['brute'],
                      'n_neighbors': k_range}
        log('Finding best alpha on ptc_mr', DIR)
        for i, cur_kernel in enumerate(self.kernelized_data_training):
            clf_knn = KNeighborsClassifier(metric='precomputed')

            # perform hyper parameter selection
            grid_search = GridSearchCV(clf_knn, param_grid, cv=10, scoring='accuracy', return_train_score=False,
                                       verbose=1)
            grid_search.fit(cur_kernel, np.ravel(self.y_train))
            log(f'Completed knn gridsearch: ({i + 1}/{len(self.kernelized_data_training)}) ', DIR)

            # construct, train optimal model and perform predictions
            clf_knn = KNeighborsClassifier(algorithm=grid_search.best_params_['algorithm'],
                                           n_neighbors=grid_search.best_params_['n_neighbors'],
                                           metric='precomputed')

            clf_knn.fit(cur_kernel, np.ravel(self.y_train))
            score = clf_knn.score(cur_kernel, np.ravel(self.y_train))
            if score > prev_score:
                prev_score = score
                best_knn = clf_knn
                best_kernel_index = i
                best_grid_search = grid_search
        log(f'finished knn fitting and found best alpha {np.arange(0.05, 1.0, 0.1)[best_kernel_index]}')
        append_hyperparams_file(False, best_grid_search, best_knn, self.dataset_name, DIR, ref=True)
        predictions = best_knn.predict(self.kernelized_data_test[best_kernel_index])
        test_accuracy = accuracy_score(self.y_test, predictions) * 100
        save_preds(predictions, self.y_test, type(best_knn).__name__, self.dataset_name, False, ref=True)
        return test_accuracy

    def predict_svm(self):
        """train and predict with svm"""
        best_kernel_index = 0
        prev_score = 0
        best_alpha = 0
        small_param_grid = {'C': [0.1, 1, 10],
                            'gamma': [0.1, 1, 10]}

        # find best alpha with less extensive param_grid
        for i, cur_kernel in enumerate(self.kernelized_data_training):
            alpha = np.arange(0.05, 1.0, 0.1)[i]
            clf_svm = svm.SVC(kernel='precomputed')

            # perform hyper parameter selection
            grid_search = GridSearchCV(clf_svm, small_param_grid, cv=5, scoring='accuracy', error_score='raise',
                                       return_train_score=False, verbose=1)
            log(f'Completing svm girdsearch with small param grid ({i + 1}/{len(self.kernelized_data_training)})', DIR)
            grid_search.fit(cur_kernel, np.ravel(self.y_train))

            # construct, train optimal model and perform predictions
            clf_svm = SVC(C=grid_search.best_params_['C'],
                          gamma=grid_search.best_params_['gamma'],
                          kernel='precomputed')

            clf_svm.fit(cur_kernel, np.ravel(self.y_train))
            score = clf_svm.score(cur_kernel, np.ravel(self.y_train))
            if score > prev_score:
                prev_score = score
                best_kernel_index = i
                best_alpha = alpha
        log(f'Completed small svm girdsearch with small param grid and found {best_alpha}', DIR)

        # do more detailed optimization now that we have the best kernel (alpha)
        big_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                          'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
        clf_svm = svm.SVC(kernel='precomputed')
        detailed_grid_search = GridSearchCV(clf_svm, big_param_grid, cv=10, scoring='accuracy', error_score='raise',
                                            return_train_score=False, verbose=1)
        log(f'Completing svm girdsearch with detailed param grid.', DIR)

        detailed_grid_search.fit(self.kernelized_data_training[best_kernel_index], np.ravel(self.y_train))
        log('finished detailed svm gridsearch', DIR)
        append_hyperparams_file(False, f"best alpha: {best_alpha}", clf_svm, self.dataset_name, DIR, ref=True)
        append_hyperparams_file(False, detailed_grid_search, clf_svm, self.dataset_name, DIR, ref=True)

        # do final fitting with best params
        best_svm = SVC(C=detailed_grid_search.best_params_['C'],
                       gamma=detailed_grid_search.best_params_['gamma'],
                       kernel='precomputed')
        log('fiting best_svm', DIR)
        best_svm.fit(self.kernelized_data_training[best_kernel_index], np.ravel(self.y_train))
        predictions = best_svm.predict(self.kernelized_data_test[best_kernel_index])
        test_accuracy = accuracy_score(self.y_test, predictions) * 100
        save_preds(predictions, self.y_test, type(best_svm).__name__, self.dataset_name, False, ref=True)

        return test_accuracy

    def predict_ann(self, device):
        return 100.0, 3.0, 2.0


def create_custom_metric(test, train, alpha):
    rows = len(test)
    cols = len(train)
    res_mat = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(i, cols):
            res_mat[i, j] = graph_edit_distance(test[i], train[j], alpha)
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
    dataset_name = "PTC_MR"

    reference_classifier = ReferenceClassifier(dataset_name)
    svm_acc = reference_classifier.predict_svm()
    knn_acc = reference_classifier.predict_knn()
    print(svm_acc)
    print(knn_acc)

    append_accuracies_file(dataset_name, "knn", False, knn_acc, DIR, ref=True)
    append_accuracies_file(dataset_name, "svm", False, svm_acc, DIR, ref=True)
