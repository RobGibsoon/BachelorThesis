from itertools import chain, combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from torch_geometric.utils import to_dense_adj, degree

feature_names = {0: "balaban", 1: "estrada", 2: "narumi", 3: "padmakar-ivan", 4: "polarity-nr", 5: "randic",
                 6: "szeged", 7: "wiener", 8: "zagreb", 9: "nodes", 10: "edges", 11: "schultz"}
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
    26: ("MUTAG", "svm", None, True)
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

NP_SEED = 42

np.random.seed(NP_SEED)


def get_degrees(graph):
    edge_index = graph.edge_index[0]
    num_nodes = graph.num_nodes
    return degree(edge_index, num_nodes)


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


def append_features_file(clf, features, count, dn):
    Path(f"log/features/").mkdir(parents=True, exist_ok=True)
    with open('log/features/features.txt', mode='a') as file:
        file.write(f"The {count} optimal features selected for {type(clf).__name__} on {dn} were: {features}\n")
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
