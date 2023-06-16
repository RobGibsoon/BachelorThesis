import csv
from datetime import datetime
from itertools import chain, combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from torch_geometric.utils import to_dense_adj, degree

feature_names = {0: "balaban", 1: "estrada", 2: "narumi", 3: "padmakar-ivan", 4: "polarity-nr", 5: "randic",
                 6: "szeged", 7: "wiener", 8: "zagreb", 9: "nodes", 10: "edges", 11: "schultz", 12: "mod_zagreb",
                 13: "hyp_wiener", 14: "n_impurity", 15: "label_entropy", 16: "edge_strength"}
inputs = {
    0: ("PTC_MR", "ann", True, False),
    1: ("PTC_MR", "knn", True, False),
    2: ("PTC_MR", "svm", True, False),
    3: ("Mutagenicity", "ann", True, False),
    4: ("Mutagenicity", "knn", True, False),
    5: ("Mutagenicity", "svm", True, False),
    6: ("MUTAG", "ann", True, False),
    7: ("MUTAG", "knn", True, False),
    8: ("MUTAG", "svm", True, False),
    9: ("PTC_MR", "ann", False, False),
    10: ("PTC_MR", "knn", False, False),
    11: ("PTC_MR", "svm", False, False),
    12: ("Mutagenicity", "ann", False, False),
    13: ("Mutagenicity", "knn", False, False),
    14: ("Mutagenicity", "svm", False, False),
    15: ("MUTAG", "ann", False, False),
    16: ("MUTAG", "knn", False, False),
    17: ("MUTAG", "svm", False, False),

    19: ("PTC_MR", "knn", None, True),
    20: ("PTC_MR", "svm", None, True),
    22: ("Mutagenicity", "knn", None, True),
    23: ("Mutagenicity", "svm", None, True),
    25: ("MUTAG", "knn", None, True),
    26: ("MUTAG", "svm", None, True),

    36: ("PTC_FM", "ann", True, False),
    37: ("PTC_FM", "knn", True, False),
    38: ("PTC_FM", "svm", True, False),
    39: ("PTC_FM", "ann", False, False),
    40: ("PTC_FM", "knn", False, False),
    41: ("PTC_FM", "svm", False, False),
    42: ("PTC_FM", "ann", None, True),
    43: ("PTC_FM", "knn", None, True),
    44: ("PTC_FM", "svm", None, True),

    45: ("PTC_MM", "ann", True, False),
    46: ("PTC_MM", "knn", True, False),
    47: ("PTC_MM", "svm", True, False),
    48: ("PTC_MM", "ann", False, False),
    49: ("PTC_MM", "knn", False, False),
    50: ("PTC_MM", "svm", False, False),
    51: ("PTC_MM", "ann", None, True),
    52: ("PTC_MM", "knn", None, True),
    53: ("PTC_MM", "svm", None, True),

    63: ("PTC_FR", "ann", True, False),
    64: ("PTC_FR", "knn", True, False),
    65: ("PTC_FR", "svm", True, False),
    66: ("PTC_FR", "ann", False, False),
    67: ("PTC_FR", "knn", False, False),
    68: ("PTC_FR", "svm", False, False),
    69: ("PTC_FR", "ann", None, True),
    70: ("PTC_FR", "knn", None, True),
    71: ("PTC_FR", "svm", None, True),

    72: ("DHFR_MD", "ann", True, False),
    73: ("DHFR_MD", "knn", True, False),
    74: ("DHFR_MD", "svm", True, False),
    75: ("DHFR_MD", "ann", False, False),
    76: ("DHFR_MD", "knn", False, False),
    77: ("DHFR_MD", "svm", False, False),
    78: ("DHFR_MD", "ann", None, True),
    79: ("DHFR_MD", "knn", None, True),
    80: ("DHFR_MD", "svm", None, True),
    81: ("DHFR_MD", "svm", None, True),

    82: ("ER_MD", "ann", True, False),
    83: ("ER_MD", "knn", True, False),
    84: ("ER_MD", "svm", True, False),
    85: ("ER_MD", "ann", False, False),
    86: ("ER_MD", "knn", False, False),
    87: ("ER_MD", "svm", False, False),
    88: ("ER_MD", "ann", None, True),
    89: ("ER_MD", "knn", None, True),
    90: ("ER_MD", "svm", None, True)
}

top_5_mRMR_features = {"PTC_MR": [14, 2, 15, 1, 0],
                       "PTC_MM": [14, 2, 15, 1, 0],
                       "PTC_FM": [14, 2, 16, 15, 8],
                       "PTC_FR": [14, 0, 1, 2, 15],
                       "MUTAG": [8, 0, 10, 5, 9],
                       "Mutagenicity": [0, 13, 16, 15, 14],
                       "ER_MD": [5, 15, 12, 0, 0],
                       "DHFR_MD": [12, 15, 9, 5, 0]
                       }
sfs_features = {
    "PTC_MR": [10, 3, 7, 9, 4, 8, 1, 12, 0, 2],
    "PTC_MM": [0, 1, 8, 3, 2, 16, 4, 5, 14, 6],
    "PTC_FM": [1, 0, 8, 9, 10, 16, 2, 12, 4, 13],
    "PTC_FR": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "MUTAG": [5, 0, 2, 8, 4, 9, 14, 12, 6, 3],
    "Mutagenicity": [16, 9, 8, 1, 15, 7, 10, 5, 11, 14],
    "ER_MD": [0, 1, 2, 3, 7, 10, 14, 12, 4, 11],
    "DHFR_MD": [2, 0, 1, 3, 4, 8, 5, 10, 6, 12]
}


def mRMR_applied_datasets(X_train, X_test, dataset_name):
    """helper method for embedding_classifier, returns the modified X_train and X_tests"""
    X_train_fs = X_train[:, top_5_mRMR_features[dataset_name]]
    X_test_fs = X_test[:, top_5_mRMR_features[dataset_name]]
    return X_train_fs, X_test_fs


def sfs_applied_datasets(X_train, X_test, dataset_name):
    """helper method for embedding_classifier, returns the modified X_train and X_tests"""
    X_train_fs = X_train[:, sfs_features[dataset_name]]
    X_test_fs = X_test[:, sfs_features[dataset_name]]
    return X_train_fs, X_test_fs


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
MOD_ZAGREB = 12
HYP_WIENER = 13
N_IMPURITY = 14
LABEL_ENTROPY = 15
EDGE_STRENGTH = 16

NP_SEED = 42

np.random.seed(NP_SEED)


def get_degrees(graph):
    edge_index = graph.edge_index[0]
    num_nodes = graph.num_nodes
    return degree(edge_index, num_nodes)


def get_csv_idx_split(dn, idx_type):
    file = open(f"log/index_splits/{dn}_{idx_type}_split.csv", "r")
    idx_split = list(csv.reader(file, delimiter=','))
    parsed_idx_split = [int(elt) for elt in idx_split[0]]
    return parsed_idx_split


def is_connected(graph):
    """this returns True if the graph is connected and False otherwise"""
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    G.add_edges_from(graph.edge_index.t().tolist())

    # Check if the graph is connected
    connected = nx.is_connected(G)
    return connected


def log(text, dir):
    """used to print all the print statements into a log.txt file"""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    Path(f"log/{dir}").mkdir(parents=True, exist_ok=True)
    with open(f'log/{dir}/log.txt', mode='a') as file:
        file.write(current_time + ": " + text + "\n")
    file.close()


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
    """returns all possible combinations of sets of indices"""
    return chain(*map(lambda x: combinations(indices_list, x), range(1, len(indices_list) + 1)))


def get_feature_names(feature_subset):
    """returns the names of the features for a list with indices 0-11"""
    count = len(feature_subset)
    assert count > 0
    features = ''
    for i, feature in enumerate(feature_subset):
        if i + 1 != len(feature_subset):
            features += (feature_names[feature])
            features += ", "
        else:
            features += (feature_names[feature])
    return features, count


def append_features_file(text):
    Path(f"log/features/").mkdir(parents=True, exist_ok=True)
    with open('log/features/features.txt', mode='a') as file:
        file.write(text)
    file.close()


def append_accuracies_file(dn, clf, fs, acc, dir, index="", ref=False):
    Path(f"log/accuracies/").mkdir(parents=True, exist_ok=True)
    if not ref:
        with open('log/accuracies/accuracies.txt', mode='a') as file:
            file.write(f'Accuracy for {dn} {clf}{index} fs={fs}: {acc}\n')
        file.close()
        log(f'Accuracy for {dn} {type(clf).__name__} fs={fs}: {acc}\n', dir)
    else:
        with open('log/accuracies/reference_accuracies.txt', mode='a') as file:
            file.write(f'Reference accuracy for {dn} {clf}: {acc}\n')
        file.close()


def append_hyperparams_file(fs, gs, clf, dn, dir, ref=False):
    Path(f"log/hyperparameters/").mkdir(parents=True, exist_ok=True)
    if not ref:
        with open('log/hyperparameters/hyperparameters.txt', mode='a') as file:
            file.write(f"The optimal hyperparameters selected for {type(clf).__name__} on {dn} and fs = "
                       f"{fs} were: {gs.best_params_}\n")
        file.close()
    else:
        with open('log/hyperparameters/reference_hyperparameters.txt', mode='a') as file:
            try:
                best_params = gs.best_params_
            except AttributeError:
                # if gs is just a string, then best_params is that string
                best_params = gs
            file.write(f"The optimal reference hyperparameters selected for {type(clf).__name__} on {dn} "
                       f"were: {best_params}\n")
        file.close()


def save_preds(preds, labels, clf, dn, fs, ref=False):
    """saves labels and predictions to a csv-file"""
    Path(f"log/predictions/").mkdir(parents=True, exist_ok=True)
    if not ref:
        data = {"preds": np.ravel(preds), "labels": np.ravel(labels)}
        df = pd.DataFrame(data)
        df.to_csv(f'log/predictions/preds_labels_{clf}_{dn}_fs{fs}.csv', index=False)
    else:
        data = {"preds": np.ravel(preds), "labels": np.ravel(labels)}
        df = pd.DataFrame(data)
        df.to_csv(f'log/predictions/reference_preds_labels_{clf}_{dn}.csv', index=False)
