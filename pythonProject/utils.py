from itertools import chain, combinations

import networkx as nx
import numpy as np
import torch
from scipy.sparse import csr_matrix
# from torch import combinations
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
    with open(f'log/{dir}/log.txt', mode='a') as file:
        file.write(text+"\n")
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
    assert len(feature_subset) > 0
    features = ''
    for i, feature in enumerate(feature_subset):
        if i + 1 != len(feature_subset):
            features += (feature_names[feature])
            features += ", "
        else:
            features += (feature_names[feature])
    return features
