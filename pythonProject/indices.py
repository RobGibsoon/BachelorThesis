import numpy as np
import torch
from numpy.linalg import LinAlgError
from utils import get_distance_matrix, get_degrees
from torch_geometric.utils import to_dense_adj


def create_zagreb_index(graph):
    """The zagreb index is the sum of the squared degrees of all non-hydrogen atoms of a molecule."""
    assert graph.is_undirected()
    degrees = get_degrees(graph)
    squared_degrees = torch.square(degrees)
    zagreb_index = torch.sum(squared_degrees).item()
    assert zagreb_index % 1 == 0
    return np.array([int(zagreb_index)])


def create_narumi_index(graph):
    """The narumi index is the product of the node-degrees of all non-hydrogen atoms."""
    assert graph.is_undirected()
    degrees = get_degrees(graph)
    narumi_index = np.prod(degrees.numpy())
    return np.array([int(narumi_index)])


def create_polarity_nr_index(graph):
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
    assert graph.is_undirected()
    adj = to_dense_adj(graph.edge_index).numpy()
    try:
        eigenvalues = np.squeeze(np.linalg.eig(adj)[0], axis=0)
    except LinAlgError:
        print(LinAlgError.__name__)
        return np.array([0])  # todo: is there a different option than just returning 0?

    estrada_index = 0
    for eigenvalue in eigenvalues:
        estrada_index += np.exp(eigenvalue)
    return np.array([estrada_index])


def create_balaban_index(graph):
    """balaban-j-index as defined by:  Alexandru T. Balaban: Highly Discriminating Distance-based Topological Index"""
    assert graph.is_undirected()
    shortest_dist_mat = get_distance_matrix(graph)
    dist_sums = np.sum(shortest_dist_mat, axis=1)
    num_nodes = graph.num_nodes
    num_edges = int(len(graph.edge_index[1]) / 2)
    num_cycles = num_edges-num_nodes+1  # according to wikipedia

    graph.coalesce()  # sort edge_index
    edges = graph.edge_index.t().tolist()
    sum_dist_neighbours = 0
    while len(edges) != 0:
        s1 = dist_sums[edges[0][0]]
        s2 = dist_sums[edges[0][1]]
        sum_dist_neighbours += 1/np.sqrt(s1*s2)
        remove_first_same_edges(edges)

    return np.array([num_edges/(num_cycles+1)*sum_dist_neighbours])


def remove_first_same_edges(edges):
    idx1 = int(edges[0][0])
    idx2 = int(edges[0][1])
    edges.remove([idx1, idx2])
    edges.remove([idx2, idx1])


def create_szeged_index(graph):
    """szeged index is the sum over each edge (u,v) of the product n1(u)*n2(v)
    where n1(u) is the number of vertices closer to u and n2 respectively closer to v"""
    assert graph.is_undirected()
    shortest_dist_mat = get_distance_matrix(graph)
    edges = graph.edge_index.t().tolist()
    szeged_index = 0
    while len(edges) != 0:
        u = edges[0][0]
        v = edges[0][1]
        n1, n2 = number_vertices_closer_to_uv(u, v, shortest_dist_mat)
        szeged_index += n1*n2
        remove_first_same_edges(edges)
    return np.array([szeged_index])


def number_vertices_closer_to_uv(u, v, shortest_dist_mat):
    n1 = 0
    n2 = 0
    # todo: maybe i can improve this by not using for loop
    for w in range(len(shortest_dist_mat)):
        if shortest_dist_mat[w][u] < shortest_dist_mat[w][v]:
            n1 += 1
        if shortest_dist_mat[w][u] > shortest_dist_mat[w][v]:
            n2 += 1

    return n1, n2


def create_padmakar_ivan_index(graph):
    """padmakar-ivan index is the sum over each edge (u,v) of the product n1(u)*n2(v)
        where n1(u) is the number of vertices closer to u and n2 respectively closer to v"""
    assert graph.is_undirected()
    shortest_dist_mat = get_distance_matrix(graph)
    edges = graph.edge_index.t().tolist()
    padmakar_ivan_index = 0
    while len(edges) != 0:
        u = edges[0][0]
        v = edges[0][1]
        n1, n2 = number_vertices_closer_to_uv(u, v, shortest_dist_mat)
        padmakar_ivan_index += n1 + n2
        remove_first_same_edges(edges)
    return np.array([padmakar_ivan_index])


