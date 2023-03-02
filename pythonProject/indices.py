import numpy as np
import torch
from utils import torch_to_csr, get_distance_matrix, get_degrees
from scipy.sparse import csr_matrix
from torch_geometric.utils import degree, dense_to_sparse, to_dense_adj
from scipy.sparse.csgraph import dijkstra


def create_zagreb_index(graph):
    """The zagreb index is the sum of the squared degrees of all non-hydrogen atoms of a molecule."""
    assert graph.is_undirected()
    degrees = get_degrees(graph)
    squared_degrees = torch.square(degrees)
    zagreb_index = torch.sum(squared_degrees).item()
    assert zagreb_index % 1 == 0
    return np.array([int(zagreb_index)])


def create_basic_descriptors(graph):
    """Gets basic descriptors like num_nodes and num_edges of a graph."""
    assert graph.is_undirected()
    num_nodes = graph.num_nodes
    num_edges = int(len(graph.edge_index[1]) / 2)
    return np.array([num_nodes, num_edges])


def create_narumi_index(graph):
    """The narumi index is the product of the node-degrees of all non-hydrogen atoms."""
    assert graph.is_undirected()
    degrees = get_degrees(graph)
    narumi_index = np.prod(degrees.numpy())
    return np.array([int(narumi_index)])


def create_polarity_number_index(graph):
    """The polarity number (a.k.a. Wiener polarity index) is the number of unordered pairs of vertices lying at
    distance 3 in a graph."""
    assert graph.is_undirected()
    dist_matrix = get_distance_matrix(graph)
    count_dist_3 = np.count_nonzero(dist_matrix == 3)
    assert count_dist_3 % 2 == 0  # since we look at an undirected graph we always expect pairs of nodes to have
    # distance 3
    polarity_nr = count_dist_3 / 2
    return np.array([int(polarity_nr)])


def create_wiener_index(graph):
    """the wiener index is defined as the sum of the lengths of the shortest paths between all pairs of nodes in the
    graph representing non-hydrogen atoms in the molecule """
    assert graph.is_undirected()
    dist_matrix = get_distance_matrix(graph)
    wiener_index = np.sum(dist_matrix) / 2
    return np.array([int(wiener_index)])


def create_randic_index(graph):
    """the randic index is defined by the sum over all edges e=(u,v) of [1/sqrt(deg(u)*deg(v))]"""
    edge_index = graph.edge_index.t()
    degrees = get_degrees(graph)
    # todo: I may be able to improve this so i don't calculate the index for every edge since they're always double
    randic_index = 0
    for edge in edge_index:
        degree_u = degrees[edge[0]]
        degree_v = degrees[edge[1]]
        randic_index += 1 / np.sqrt(degree_u.numpy() * degree_v.numpy())

    return np.array([randic_index / 2])


def create_estrada_index(graph):
    return np.array([0])


def create_szeged_index(graph):
    return np.array([0])


def create_padmakar_ivan_index(graph):
    return np.array([0])


def create_balaban_centric_index(graph):
    return np.array([0])


def get_all_indices(graph):
    """returns a np-array with all the indices of a graph"""
    indices = create_zagreb_index(graph)
    indices = np.append(indices, create_basic_descriptors(graph))
    return indices
