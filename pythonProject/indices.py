import numpy as np
import torch
from utils import torch_to_csr
from scipy.sparse import csr_matrix
from torch_geometric.utils import degree, dense_to_sparse, to_dense_adj
from scipy.sparse.csgraph import dijkstra


def create_zagreb_index(graph):
    """The zagreb index is the sum of the squared degrees of all non-hydrogen atoms of a molecule."""
    assert graph.is_undirected()
    edge_index = graph.edge_index[0]
    num_nodes = graph.num_nodes

    degrees = degree(edge_index, num_nodes)
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
    edge_index = graph.edge_index[0]
    num_nodes = graph.num_nodes

    degrees = degree(edge_index, num_nodes)
    narumi_index = np.prod(degrees.numpy())
    return np.array([int(narumi_index)])


def create_polarity_number_index(graph):
    """The polarity number (a.k.a. Wiener polarity index) is the number of unordered pairs of vertices lying at
    distance 3 in a graph."""
    dist_matrix = get_distance_matrix(graph)
    count_dist_3 = np.count_nonzero(dist_matrix == 3)
    assert count_dist_3 % 2 == 0  # since we look at an undirected graph we always expect pairs of nodes to have distance 3
    polarity_nr = count_dist_3 / 2
    return np.array([int(polarity_nr)])


def get_distance_matrix(graph):
    adj = to_dense_adj(graph.edge_index)
    graph = np.squeeze(adj.numpy(), axis=0)  # collapse outer dimension to get (n,n) array
    graph = graph.tolist()
    if len(graph) == 0:
        return np.array([0])
    csr_mat = csr_matrix(graph)
    dist_matrix = dijkstra(csgraph=csr_mat, directed=False)
    return dist_matrix


def create_wiener_index(graph):
    return np.array([0])


def create_randic_index(graph):
    return np.array([0])


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
