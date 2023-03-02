import numpy as np
from scipy.sparse import csr_matrix


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
