import numpy as np
import torch
from numpy.linalg import LinAlgError
from utils import get_distance_matrix, get_degrees
from torch_geometric.utils import to_dense_adj


def create_zagreb_index(graph):
    """The zagreb index is the sum of the squared degrees of all non-hydrogen atoms of a molecule.
    DOI 10.1007/s11464-015-0431-9"""
    assert graph.is_undirected()
    degrees = get_degrees(graph)
    squared_degrees = torch.square(degrees)
    zagreb_index = torch.sum(squared_degrees).item()
    assert zagreb_index % 1 == 0
    return np.array([int(zagreb_index)])


def create_narumi_index(graph):
    """The narumi index is the product of the node-degrees of all non-hydrogen atoms.
    https://doi.org/10.1016/j.aml.2011.12.018"""
    assert graph.is_undirected()
    degrees = get_degrees(graph)
    narumi_index = np.prod(degrees.numpy())
    return np.array([int(narumi_index)])


def create_polarity_nr_index(graph):
    """The polarity number (a.k.a. Wiener polarity index) is the number of unordered pairs of vertices lying at
    distance 3 in a graph.
    DOI 10.1002/qua.26627"""
    assert graph.is_undirected()
    dist_matrix = get_distance_matrix(graph)
    count_dist_3 = np.count_nonzero(dist_matrix == 3)
    assert count_dist_3 % 2 == 0  # since we look at an undirected graph we always expect pairs of nodes to have
    # distance 3
    polarity_nr = count_dist_3 / 2
    return np.array([int(polarity_nr)])


def create_wiener_index(graph):
    """the wiener index is defined as the sum of the lengths of the shortest paths between all pairs of nodes in the
    graph representing non-hydrogen atoms in the molecule
    https://doi.org/10.1016/j.dam.2012.01.014"""
    assert graph.is_undirected()
    dist_matrix = get_distance_matrix(graph)
    wiener_index = np.sum(dist_matrix) / 2
    return np.array([int(wiener_index)])


def create_randic_index(graph):
    """the randic index is defined by the sum over all edges e=(u,v) of [1/sqrt(deg(u)*deg(v))]
    https://doi.org/10.1016/j.akcej.2017.09.006"""
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
    """
    estrada index of a graph with n vertices and l_i being the n eigenvalues is defined as
    the sum_i=1^n e^(l_i)
    https://doi.org/10.1016/j.laa.2007.06.020
    """
    assert graph.is_undirected()
    adj = to_dense_adj(graph.edge_index).numpy()
    try:
        eigenvalues = np.squeeze(np.linalg.eig(adj)[0], axis=0)
    except LinAlgError:
        print(LinAlgError.__name__)
        return np.array([float("NaN")])

    estrada_index = 0
    for eigenvalue in eigenvalues:
        estrada_index += np.exp(eigenvalue)
    return np.array([estrada_index])


def remove_same_edges(edges, idx1, idx2):
    if not isinstance(edges, list):
        edges = edges.tolist()
    edges.remove([idx1, idx2])
    edges.remove([idx2, idx1])


def create_balaban_index(graph):
    """balaban-j-index as defined by:  Alexandru T. Balaban: Highly Discriminating Distance-based Topological Index
    https://doi.org/10.1016/0009-2614(82)80009-2"""
    assert graph.is_undirected()
    shortest_dist_mat = get_distance_matrix(graph)
    dist_sums = np.sum(shortest_dist_mat, axis=1)
    num_nodes = graph.num_nodes
    num_edges = int(len(graph.edge_index[1]) / 2)
    num_cycles = num_edges - num_nodes + 1  # according to wikipedia

    graph.coalesce()  # sort edge_index
    edges = graph.edge_index.t().tolist()
    sum_dist_neighbours = 0
    while len(edges) != 0:
        s1 = dist_sums[edges[0][0]]
        s2 = dist_sums[edges[0][1]]
        sum_dist_neighbours += 1 / np.sqrt(s1 * s2)
        idx1 = int(edges[0][0])
        idx2 = int(edges[0][1])
        remove_same_edges(edges, idx1, idx2)

    return np.array([num_edges / (num_cycles + 1) * sum_dist_neighbours])


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


def create_szeged_index(graph):
    """szeged index is the sum over each edge (u,v) of the product n1(u)*n2(v)
    where n1(u) is the number of vertices closer to u and n2 respectively closer to v
    DOI 10.1166/jctn.2011.1681"""
    assert graph.is_undirected()
    shortest_dist_mat = get_distance_matrix(graph)
    edges = graph.edge_index.t().tolist()
    szeged_index = 0
    while len(edges) != 0:
        u = edges[0][0]
        v = edges[0][1]
        n1, n2 = number_vertices_closer_to_uv(u, v, shortest_dist_mat)
        szeged_index += n1 * n2
        idx1 = int(edges[0][0])
        idx2 = int(edges[0][1])
        remove_same_edges(edges, idx1, idx2)
    return np.array([szeged_index])


def number_edges_closer_to_uv(edge, edges, shortest_dist_mat):
    n1, n2 = 0, 0
    u, v = edge[0], edge[1]
    copy = edges.copy()
    remove_same_edges(copy, u, v)
    copy = np.array(copy)
    for edge_iter in copy:
        a, b = edge_iter[0], edge_iter[1]
        if shortest_dist_mat[a, u] == shortest_dist_mat[a, v] and \
                shortest_dist_mat[b, u] == shortest_dist_mat[b, v]:
            continue
        if min(shortest_dist_mat[a, u], shortest_dist_mat[b, u]) < \
                min(shortest_dist_mat[a, v], shortest_dist_mat[b, v]):
            n1 += 1
        if min(shortest_dist_mat[a, v], shortest_dist_mat[b, v]) < \
                min(shortest_dist_mat[a, u], shortest_dist_mat[b, u]):
            n2 += 1
    return n1 / 2, n2 / 2


def create_padmakar_ivan_index(graph):
    """padmakar-ivan index is the sum over each edge (u,v) of the product n1(u)*n2(v)
        where n1(u) is the number of edges closer to u and n2 respectively closer to v
        DOI 10.1166/jctn.2011.1681"""
    assert graph.is_undirected()
    shortest_dist_mat = get_distance_matrix(graph)
    full_edges = graph.edge_index.t().tolist()
    edges = full_edges.copy()
    padmakar_ivan_index = 0
    while len(edges) != 0:
        edge = edges[0]
        n1, n2 = number_edges_closer_to_uv(edge, full_edges, shortest_dist_mat)
        padmakar_ivan_index += n1 + n2

        idx1 = int(edges[0][0])
        idx2 = int(edges[0][1])
        remove_same_edges(edges, idx1, idx2)
    return np.array([int(padmakar_ivan_index)])


def create_schultz_index(graph):
    """The schultz index is the over n vertices: sum_i^n sum_j^n deg(i)*(A_ij+dist(i,j)) I calculate this like so:
    sum_i^n deg(i)* [sum_j^n (A_ij+dist(i,j))] Also since our graph is undirected the adjacency matrix is symmetrical
    and because there are no loops the A_ii and dist(i,i) entries will always be zero!

    Wiener and Schultz Molecular Topological Indices of Graphs with Specified Cut Edges by Hongbo Hua
    MATCH Commun. Math. Comput. Chem. 61 (2009) 643-651
    ISSN 0340 - 6253
    found under link: https://match.pmf.kg.ac.rs/electronic_versions/Match61/n3/match61n3_643-651.pdf"""
    assert graph.is_undirected()
    adj = to_dense_adj(graph.edge_index).numpy()
    degrees = get_degrees(graph).numpy()
    shortest_dist_mat = get_distance_matrix(graph)
    adj_short = np.sum(np.squeeze(adj + shortest_dist_mat), axis=1)
    assert adj_short.shape == degrees.shape

    schultz_index = 0
    for i in range(len(degrees)):
        schultz_index += degrees[i] * adj_short[i]

    return np.array([int(schultz_index)])
