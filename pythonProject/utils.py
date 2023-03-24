from itertools import chain

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch import combinations
from torch_geometric.utils import to_dense_adj, degree
from scipy.sparse.csgraph import dijkstra

feature_names = {0: "balaban", 1: "estrada", 2: "narumi", 3: "padmakar-ivan", 4: "polarity-nr", 5: "randic",
                 6: "szeged", 7: "wiener", 8: "zagreb", 9: "nodes", 10: "edges", 11: "schultz"}
BALABAN = 0
ESTRADA = 1
NARUMI = 2
PADMAKAR_IVAN = 3
POLARITY_NR = 4
RANDIC = 5
SZEGED = 6
WIENER = 7
ZAGREB = 8
NODES = 9
EDGES = 10
SCHULTZ = 11


def get_degrees(graph):
    edge_index = graph.edge_index[0]
    num_nodes = graph.num_nodes
    return degree(edge_index, num_nodes)


def get_distance_matrix(graph):
    """for a graph returns a np-array (num_nodes, num_nodes) with the shortest path distances for each of the nodes"""
    assert graph.is_undirected()
    adj = to_dense_adj(graph.edge_index)
    graph = np.squeeze(adj.numpy(), axis=0)  # collapse outer dimension to get (n,n) array
    graph = graph.tolist()
    if len(graph) == 0:
        return np.array([0])
    csr_mat = csr_matrix(graph)
    dist_matrix = dijkstra(csgraph=csr_mat, directed=False)
    return dist_matrix


def all_subsets(indices_list):
    return chain(*map(lambda x: combinations(torch.from_numpy(indices_list), x), range(0, len(indices_list) + 1)))


def best_subset_cv(estimator, X, y, cv=3):
    n_features = 6  # todo change to X.shape[1]
    best_score = -np.inf
    best_subset = None
    count = 1

    for subset in all_subsets(np.arange(n_features)):
        score = cross_val_score(estimator, np.reshape(X[:, subset], (275, -1)), y, cv=cv).mean()
        if score > best_score:
            best_score, best_subset = score, subset
        print(f'subset {count}/{2 ** n_features - 1}')
        count += 1

    print(f"best_subset: {best_subset.numpy()} with best score: {best_score}")
    return best_subset.numpy(), best_score


def feature_selected_sets(clf, X_train, X_test, y):
    if not isinstance(clf, (KNeighborsClassifier, SVC)):
        return X_train, X_test
    best_subset, best_score = best_subset_cv(clf, X_train, y)
    features = get_feature_names(best_subset)
    print(f"The optimal features selected for {type(clf).__name__} were: {features}")
    X_train_fs = X_train[:, best_subset]
    X_test_fs = X_test[:, best_subset]
    assert (X_train_fs.shape[0], len(best_subset)) == X_train_fs.shape
    return X_train_fs, X_test_fs


def get_feature_names(feature_subset):
    features = ''
    for i, feature in enumerate(feature_subset):
        if i != len(feature_subset):
            features += (feature_names[feature])
            features += ", "
        else:
            features += (feature_names[feature])
    return features
