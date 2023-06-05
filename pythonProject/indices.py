import numpy as np
import torch
from numpy.linalg import LinAlgError
from scipy.stats import entropy
from torch_geometric.utils import to_dense_adj

from utils import get_distance_matrix, get_degrees


def zagreb_index(graph):
    """The zagreb index is the sum of the squared degrees of all non-hydrogen atoms of a molecule.
    DOI https://doi.org/10.1007/s11464-015-0431-9"""
    assert graph.is_undirected()
    degrees = get_degrees(graph)
    squared_degrees = torch.square(degrees)
    zagreb_index = torch.sum(squared_degrees).item()
    assert zagreb_index % 1 == 0
    return np.array(int(zagreb_index))


def narumi_index(graph, dataset_name):
    """The narumi index is the product of the node-degrees of all non-hydrogen atoms.
    https://doi.org/10.1016/j.aml.2011.12.018"""
    assert graph.is_undirected()
    degrees = get_degrees(graph).numpy()
    if dataset_name == "DHFR" or dataset_name == "ER_MD":
        # an overflow is occasionally generated on these datasets, therefore 8 random degreees are chosen to
        # calculate the product of the degrees
        t = 8
        np.random.shuffle(degrees)
        degrees = degrees[:t]

    narumi_index = np.prod(degrees)

    return np.array(float(narumi_index))


def polarity_nr_index(graph):
    """The polarity number (a.k.a. Wiener polarity index) is the number of unordered pairs of vertices lying at
    distance 3 in a graph.
    DOI https://doi.org/10.1002/qua.26627"""
    assert graph.is_undirected()
    dist_matrix = get_distance_matrix(graph)
    count_dist_3 = np.count_nonzero(dist_matrix == 3)
    assert count_dist_3 % 2 == 0  # since we look at an undirected graph we always expect pairs of nodes to have
    # distance 3
    polarity_nr = count_dist_3 / 2
    return np.array(int(polarity_nr))


def wiener_index(graph):
    """the wiener index is defined as the sum of the lengths of the shortest paths between all pairs of nodes in the
    graph representing non-hydrogen atoms in the molecule
    https://doi.org/10.1016/j.dam.2012.01.014"""
    assert graph.is_undirected()
    dist_matrix = get_distance_matrix(graph)
    wiener_index = np.sum(dist_matrix) / 2
    return np.array(int(wiener_index))


def randic_index(graph):
    """the randic index is defined by the sum over all edges e=(u,v) of [1/sqrt(deg(u)*deg(v))]
    https://doi.org/10.1016/j.akcej.2017.09.006"""
    edge_index = graph.edge_index.t()
    degrees = get_degrees(graph)
    degree_u = degrees[edge_index.T[0]]
    degree_v = degrees[edge_index.T[1]]
    randic_index = torch.sum(1 / torch.sqrt(torch.mul(degree_u, degree_v)))
    return np.array(randic_index / 2)


def estrada_index(graph):
    """
    estrada index of a graph with n vertices and l_i being the n eigenvalues is defined as
    the sum_i=1^n e^(l_i)
    https://doi.org/10.1016/j.laa.2007.06.020
    It can be estimated as seen in https://doi.org/10.1016/j.cplett.2007.08.053
    """
    assert graph.is_undirected()
    adj = to_dense_adj(graph.edge_index).numpy()
    try:
        eigenvalues = np.squeeze(np.linalg.eigvals(adj), axis=0)
    except LinAlgError:
        print(LinAlgError.__name__)
        return np.array([float("NaN")])
    estrada_index = np.sum(np.exp(eigenvalues))

    if np.iscomplexobj(estrada_index):
        num_edges = int(len(graph.edge_index[1]) / 2)
        num_nodes = graph.num_nodes
        k = np.sqrt(6 * num_edges / num_nodes)
        estrada_index = num_nodes / 2 * (np.exp(k) - np.exp(-k) / k)
        assert not np.iscomplex(estrada_index)
    return np.array(estrada_index)


def remove_same_edges(edges, idx1, idx2):
    if not isinstance(edges, list):
        edges = edges.tolist()
    edges.remove([idx1, idx2])
    edges.remove([idx2, idx1])


def balaban_index(graph):
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

    return np.array(num_edges / (num_cycles + 1) * sum_dist_neighbours)


def szeged_index(graph):
    """szeged index is the sum over each edge (u,v) of the product n1(u)*n2(v)
    where n1(u) is the number of vertices closer to u and n2 respectively closer to v
    DOI https://doi.org/10.1166/jctn.2011.1681"""
    assert graph.is_undirected()
    shortest_dist_mat = get_distance_matrix(graph)
    edges = graph.edge_index.t()
    n1 = np.sum(shortest_dist_mat[edges[:, 0]] < shortest_dist_mat[edges[:, 1]], axis=1)
    n2 = np.sum(shortest_dist_mat[edges[:, 0]] > shortest_dist_mat[edges[:, 1]], axis=1)
    szeged_index = np.sum(n1 * n2)
    return np.array(szeged_index / 2)


def number_edges_closer_to_uv(edge, edges, shortest_dist_mat):
    n1, n2 = 0, 0
    u, v = edge[0], edge[1]
    copy = edges.copy()
    copy.discard(edge)
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
    return n1, n2


def padmakar_ivan_index(graph):
    """padmakar-ivan index is the sum over each edge (u,v) of the product n1(u)*n2(v)
        where n1(u) is the number of edges closer to u and n2 respectively closer to v
        DOI https://doi.org/10.1166/jctn.2011.1681"""
    assert graph.is_undirected()
    shortest_dist_mat = get_distance_matrix(graph)
    edges = set(tuple(sorted(edge)) for edge in graph.edge_index.t().tolist())
    padmakar_ivan_index = 0
    for edge in edges:
        n1, n2 = number_edges_closer_to_uv(edge, edges, shortest_dist_mat)
        padmakar_ivan_index += n1 + n2
    return np.array(int(padmakar_ivan_index))


def schultz_index(graph):
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
    schultz_index = np.sum(degrees * adj_short)
    return np.array(int(schultz_index))


def modified_zagreb_index(graph):
    """defined in Ranjini, P.S., Lokesha, V., Usha, A., 2013. Relation between
    phenylene and hexagonal squeez using harmonic index. Int. J.
    Graph Theory 1, 116–121.
    Mentioned in https://doi.org/10.1016/j.arabjc.2020.05.021"""
    assert graph.is_undirected()
    degrees = get_degrees(graph).numpy()
    edges = set(tuple(sorted(edge)) for edge in graph.edge_index.t().tolist())

    mzagreb_idx = 0
    for edge in edges:
        deg_u = degrees[edge[0]]
        deg_v = degrees[edge[1]]
        part = (deg_u + deg_v) / (deg_u * deg_v)
        mzagreb_idx += part

    return np.array(mzagreb_idx)


def hyper_wiener_index(graph):
    """as defined in https://doi.org/10.48550/arXiv.1212.4411"""
    assert graph.is_undirected()

    dist_matrix = get_distance_matrix(graph)
    w_1 = wiener_index(graph)
    w_2 = np.sum(np.square(dist_matrix)) / 2
    return np.array(1 / 2 * (w_1 + w_2))


def neighborhood_impurity(graph):
    """calculates the label entropy of a graph like in https://doi.org/10.1002/sam.11153
    it is the average impurity degree over nodes with a positive impurity
    the impurity of a node is defined as the number of neighbors with a different label
    """
    assert graph.is_undirected()

    node_labels = graph.x.argmax(dim=1).numpy()
    node_impurity_degrees = np.array([0])

    # Loop over all nodes to get node impurities
    for node in range(graph.num_nodes):
        node_impurity = 0
        cur_node_label = node_labels[node]

        # Find neighbors for current label
        neighbors_indexes = []
        for i, num in enumerate(graph.edge_index[0]):
            if num.item() == node:
                neighbors_indexes.append(i)
        neighbors = [graph.edge_index[1][i].item() for i in neighbors_indexes]

        # Calculate impurity for current node
        for neighbor in neighbors:
            if node_labels[neighbor] != cur_node_label:
                node_impurity += 1

        # update list of node impurities
        if node_impurity > 0:
            node_impurity_degrees = np.append(node_impurity_degrees, node_impurity)

    graph_impurity = np.mean(node_impurity_degrees)
    assert graph_impurity is not None
    return graph_impurity


def label_entropy(graph):
    """calculates the label entropy of a graph like in https://doi.org/10.1002/sam.11153"""
    assert graph.is_undirected()

    # Extract the node labels
    # Extract the node labels from graph.x
    node_labels = graph.x.argmax(dim=1).numpy()

    # Calculate the label probabilities
    label_frequencies = np.bincount(node_labels) / len(node_labels)

    return np.array(entropy(label_frequencies, base=2))


def avg_edge_strength(graph):
    """takes the average edge strength within a molecule"""
    assert graph.is_undirected()

    return np.mean(graph.edge_attr.argmax(dim=1).numpy())


if __name__ == "__main__":
    """dataset_name = "PROTEINS"
    data = TUDataset(root=f'/tmp/{dataset_name}', name=f'{dataset_name}')
    for i in range(len(data)):
        print(f"graph index {i} : ", modified_zagreb_index(data[i]))"""
