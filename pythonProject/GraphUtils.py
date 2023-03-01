import numpy as np
import torch
from torch_geometric.utils import degree


def get_zagreb_index(graph):
    """The zagreb index is the sum of the squared degrees of all non-hydrogen atoms of a molecule.
    It is expected to receive undirected graphs in order to calculate the degree"""
    assert graph.is_undirected()
    edge_index = graph.edge_index[0]
    num_nodes = graph.num_nodes

    degrees = degree(edge_index, num_nodes)
    squared_degrees = torch.square(degrees)
    zagreb_index = torch.sum(squared_degrees).item()
    assert zagreb_index % 1 == 0
    return np.array([int(zagreb_index)])


def get_basic_descriptors(graph):
    assert graph.is_undirected()
    num_nodes = graph.num_nodes
    num_edges = int(len(graph.edge_index[1])/2)
    return np.array([num_nodes, num_edges])


def get_all_indices(graph):
    indices = get_zagreb_index(graph)
    indices = np.add(indices, get_basic_descriptors(graph))
    return indices
