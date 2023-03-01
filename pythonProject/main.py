# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import torch
import torch_geometric
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree, to_networkx
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
class Enzymes:

    '''here is an example of how to import enzymes dataset and get batches of size 34'''
    def __init__(self):
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True).shuffle()
        data = dataset[0]
        assert data.is_undirected
        loader = DataLoader(dataset, batch_size=34, shuffle=True)

        for batch in loader:
            print(type(batch))
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def get_zagreb_index(graph):
    """the zagreb index is the sum of the squared degrees of all non-hydrogen atoms of a molecule"""
    edge_index = graph.edge_index[0]
    num_nodes = graph.num_nodes

    degrees = degree(edge_index, num_nodes)
    squared_degrees = torch.square(degrees)
    zagreb_index = torch.sum(squared_degrees).item()
    assert zagreb_index % 1 == 0
    return int(zagreb_index)
    rand = torch.rand(5, 5)
    print(rand)
    print(rand[3:])  # prints last 5-3 rows
    print(rand[:3])  # prints first 3 rows

if __name__ == "__main__":
    dataset = TUDataset(root='/tmp/PTC_MR', name='PTC_MR')

    # print(get_zagreb_index(dataset[0]))
    # print(f"y value of first graph: {int(dataset[0].y)}")
    # print(f'size of my dataset: {len(dataset)}')

    print(torch_geometric.__version__)
    data = dataset[10]
    print(type(dataset[0]))
    G = to_networkx(data)
    #G = nx.Graph()
    #G.add_edges_from(data.edge_index.t().tolist())
    nx.draw_networkx(G, pos=nx.spring_layout(G), with_labels=False)
    plt.show()
