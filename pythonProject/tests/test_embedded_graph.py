from unittest import TestCase

import torch
from torch_geometric.data import Data


class TestEmbeddedGraph(TestCase):
    edge_index_example = torch.tensor([[0, 1, 1, 2],
                                       [1, 0, 2, 1]], dtype=torch.long)
    x_example = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    data_example = Data(x=x_example, edge_index=edge_index_example)

    def test_construction(self):
        # todo: test construction
        self.assertTrue(False)
