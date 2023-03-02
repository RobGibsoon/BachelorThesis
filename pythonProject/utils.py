import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.utils import to_dense_adj, degree
from scipy.sparse.csgraph import dijkstra


def get_degrees(graph):
    edge_index = graph.edge_index[0]
    num_nodes = graph.num_nodes
    return degree(edge_index, num_nodes)


def torch_to_csr(val_data):
    x = val_data.x
    dim = len(x)
    print(dim)
    edge_index = val_data.edge_index
    print(edge_index)  # sparse tensor
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    edge_num = len(row)
    data = np.ones(edge_num)
    mtx = csr_matrix((data, (row, col)), shape=(dim, dim))


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
