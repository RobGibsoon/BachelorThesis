# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import torch
import torch_geometric
from embedded_graph import EmbeddedGraph
from indices import create_polarity_number_index
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree, to_networkx, is_sparse, to_dense_adj
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def create_embedded_graph_set(graph_set):
    embedded_graphs = []
    for i in range(len(graph_set)):
        g = EmbeddedGraph(graph_set[i])
        embedded_graphs.append(g)
        # print(g.embedding)
        # g = to_networkx(embedded_graphs[i])
        # nx.draw_networkx(g, pos=nx.spring_layout(g), with_labels=False, arrows=False)
        # plt.show()

    return embedded_graphs


def testing_stuff(data):
    edge_index_example_bigger = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 2, 5],
                                              [1, 0, 2, 1, 3, 2, 4, 3, 5, 2]], dtype=torch.long)
    x_example_bigger = torch.tensor([[-1], [0], [1], [1], [1], [1]], dtype=torch.float)

    data_directed = Data(x=x_example_bigger, edge_index=edge_index_example_bigger)
    g = to_networkx(data_directed)
    nx.draw_networkx(g, pos=nx.spring_layout(g), with_labels=False, arrows=False)
    plt.show()


if __name__ == "__main__":
    dataset = TUDataset(root='/tmp/PTC_MR', name='PTC_MR')
    test_data = dataset[1]
    g = to_networkx(test_data)
    nx.draw_networkx(g, pos=nx.spring_layout(g), with_labels=False, arrows=False)
    plt.show()
    #testing_stuff(test_data)
    testing_stuff(test_data)
