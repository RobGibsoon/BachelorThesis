import csv
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
                 13: "hyp_wiener", 14: "n_impurity", 15: "label_entropy", 16: "edge_strength", }
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
    18: ("PTC_MR", "ann", None, True),
    19: ("PTC_MR", "knn", None, True),
    20: ("PTC_MR", "svm", None, True),
    21: ("Mutagenicity", "ann", None, True),
    22: ("Mutagenicity", "knn", None, True),
    23: ("Mutagenicity", "svm", None, True),
    24: ("MUTAG", "ann", None, True),
    25: ("MUTAG", "knn", None, True),
    26: ("MUTAG", "svm", None, True),
    27: ("AIDS", "ann", False, True),
    28: ("AIDS", "knn", False, True),
    29: ("AIDS", "svm", False, True),
    30: ("AIDS", "ann", True, True),
    31: ("AIDS", "knn", True, True),
    32: ("AIDS", "svm", True, True),
    33: ("AIDS", "ann", None, True),
    34: ("AIDS", "knn", None, True),
    35: ("AIDS", "svm", None, True),
    36: ("PTC_FM", "ann", False, True),
    37: ("PTC_FM", "knn", False, True),
    38: ("PTC_FM", "svm", False, True),
    39: ("PTC_FM", "ann", True, True),
    40: ("PTC_FM", "knn", True, True),
    41: ("PTC_FM", "svm", True, True),
    42: ("PTC_FM", "ann", None, True),
    43: ("PTC_FM", "knn", None, True),
    44: ("PTC_FM", "svm", None, True),
    45: ("PTC_MM", "ann", False, True),
    46: ("PTC_MM", "knn", False, True),
    47: ("PTC_MM", "svm", False, True),
    48: ("PTC_MM", "ann", True, True),
    49: ("PTC_MM", "knn", True, True),
    50: ("PTC_MM", "svm", True, True),
    51: ("PTC_MM", "ann", None, True),
    52: ("PTC_MM", "knn", None, True),
    53: ("PTC_MM", "svm", None, True),
    54: ("PTC_MM", "ann", False, True),
    55: ("PTC_MM", "knn", False, True),
    56: ("PTC_MM", "svm", False, True),
    57: ("PTC_MM", "ann", True, True),
    58: ("PTC_MM", "knn", True, True),
    59: ("PTC_MM", "svm", True, True),
    60: ("PTC_MM", "ann", None, True),
    61: ("PTC_MM", "knn", None, True),
    62: ("PTC_MM", "svm", None, True),
    63: ("PTC_FR", "ann", False, True),
    64: ("PTC_FR", "knn", False, True),
    65: ("PTC_FR", "svm", False, True),
    66: ("PTC_FR", "ann", True, True),
    67: ("PTC_FR", "knn", True, True),
    68: ("PTC_FR", "svm", True, True),
    69: ("PTC_FR", "ann", None, True),
    70: ("PTC_FR", "knn", None, True),
    71: ("PTC_FR", "svm", None, True),
    72: ("DHFR_MD", "ann", False, True),
    73: ("DHFR_MD", "knn", False, True),
    74: ("DHFR_MD", "svm", False, True),
    75: ("DHFR_MD", "ann", True, True),
    76: ("DHFR_MD", "knn", True, True),
    77: ("DHFR_MD", "svm", True, True),
    78: ("DHFR_MD", "ann", None, True),
    79: ("DHFR_MD", "knn", None, True),
    80: ("DHFR_MD", "svm", None, True),
    81: ("DHFR_MD", "svm", None, True),
    82: ("ER_MD", "ann", False, True),
    83: ("ER_MD", "knn", False, True),
    84: ("ER_MD", "svm", False, True),
    85: ("ER_MD", "ann", True, True),
    86: ("ER_MD", "knn", True, True),
    87: ("ER_MD", "svm", True, True),
    88: ("ER_MD", "ann", None, True),
    89: ("ER_MD", "knn", None, True),
    90: ("ER_MD", "svm", None, True)
}

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
    Path(f"log/{dir}").mkdir(parents=True, exist_ok=True)
    with open(f'log/{dir}/log.txt', mode='a') as file:
        file.write(text + "\n")
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
