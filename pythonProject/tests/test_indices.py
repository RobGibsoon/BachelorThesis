import unittest
from unittest import TestCase

import networkx as nx
import numpy as np
import torch
from indices import create_zagreb_index, create_basic_descriptors, get_all_indices, create_polarity_number_index
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


class Test(TestCase):
    edge_index_example = torch.tensor([[0, 1, 1, 2],
                                       [1, 0, 2, 1]], dtype=torch.long)
    x_example = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    edge_index_empty = torch.tensor([[],
                                     []], dtype=torch.long)
    x_empty = torch.tensor([[]], dtype=torch.float)
    edge_index_directed = torch.tensor([[0, 1, 1],
                                        [0, 1, 0]], dtype=torch.long)
    x_directed = torch.tensor([[1], [2]], dtype=torch.float)
    edge_index_example_bigger = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 5, 2],
                                              [1, 0, 2, 1, 3, 2, 4, 3, 2, 5]], dtype=torch.long)
    x_example_bigger = torch.tensor([[-1], [0], [1], [1], [1], [1]], dtype=torch.float)

    data_directed = Data(x=x_directed, edge_index=edge_index_directed)
    data_empty = Data(x=x_empty, edge_index=edge_index_empty)
    data_example = Data(x=x_example, edge_index=edge_index_example)
    data_example_bigger = Data(x=x_example_bigger, edge_index=edge_index_example_bigger)

    def test_get_zagreb_index(self):
        expected = np.array([6])
        result = create_zagreb_index(self.data_example)
        self.assertTrue(np.array_equal(result, expected), f'calculating zagreb_index failed: expected {expected} but '
                                                          f'got {result}')

    def test_get_zagreb_index_empty_graph(self):
        expected = np.array([0])
        result = create_zagreb_index(self.data_empty)
        self.assertTrue(np.array_equal(result, expected), f'calculating zagreb_index failed: expected {expected} but '
                                                          f'got {result}')

    def test_get_zagreb_index_directed_graph(self):
        self.assertFalse(self.data_directed.is_undirected())
        with self.assertRaises(AssertionError):
            create_zagreb_index(self.data_directed)

    def test_get_basic_descriptors(self):
        expected = np.array([3, 2])
        result = create_basic_descriptors(self.data_example)
        self.assertTrue(np.array_equal(result, expected))

    def test_get_polarity_nr(self):
        expected = np.array([0])
        result1 = create_polarity_number_index(self.data_example)
        result2 = create_polarity_number_index(self.data_empty)
        self.assertTrue(np.array_equal(result1, expected),
                        f'getting polarity_nr failed: expected {expected} but got {result1}')
        self.assertTrue(np.array_equal(result2, expected),
                        f'getting polarity_nr failed: expected {expected} but got {result2}')
        expected2 = np.array([4])
        result3 = create_polarity_number_index(self.data_example_bigger)
        self.assertTrue(np.array_equal(result3, expected2),
                        f'getting polarity_nr failed: expected {expected2} but got {result3}')

    def test_get_all_indices(self):
        expected = np.array([6, 3, 2])
        result = get_all_indices(self.data_example)
        self.assertTrue(np.array_equal(result, expected),
                        f'getting all indices failed: expected {expected} but got {result}')


if __name__ == '__main__':
    unittest.main()
