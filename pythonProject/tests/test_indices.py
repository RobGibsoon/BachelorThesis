import unittest
from unittest import TestCase

import matplotlib
import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from indices import create_zagreb_index, create_polarity_nr_index, \
    create_wiener_index, create_randic_index, create_estrada_index, create_balaban_index, create_padmakar_ivan_index, \
    create_szeged_index, create_schultz_index, create_balaban_index_wrong

matplotlib.use('TkAgg')

"""
    use the following code to draw a graph
    # g = to_networkx(test_data)
    # nx.draw_networkx(g, pos=nx.spring_layout(g), with_labels=False, arrows=False)
    # plt.show()
    """


class Test(TestCase):
    edge_index_example = torch.tensor([[0, 1, 1, 2],
                                       [1, 0, 2, 1]], dtype=torch.long)
    x_example = torch.tensor([[1], [1], [1]], dtype=torch.float)
    edge_index_empty = torch.tensor([[],
                                     []], dtype=torch.long)
    x_empty = torch.tensor([[]], dtype=torch.float)
    edge_index_directed = torch.tensor([[0, 1, 1],
                                        [0, 1, 0]], dtype=torch.long)
    x_directed = torch.tensor([[1], [2]], dtype=torch.float)
    edge_index_example_pi = torch.tensor([[0, 1, 1, 5, 1, 2, 2, 3, 3, 4, 5, 2],
                                          [1, 0, 5, 1, 2, 1, 3, 2, 4, 3, 2, 5]], dtype=torch.long)
    edge_index_example_bigger = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 5, 2],
                                              [1, 0, 2, 1, 3, 2, 4, 3, 2, 5]], dtype=torch.long)
    x_example_bigger = torch.tensor([[1], [1], [1], [1], [1], [1]], dtype=torch.float)

    ei_balban_test = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 5, 2, 4, 7, 5, 6],
                                   [1, 0, 2, 1, 3, 2, 4, 3, 2, 5, 7, 4, 6, 5]], dtype=torch.long)
    x_balban_test = torch.tensor([[1], [1], [1], [1], [1], [1], [1], [1]], dtype=torch.float)

    ei_propane_test = torch.tensor([[0, 1, 1, 2, 3, 4, 5, 0, 0, 0, 6, 7, 1, 1, 8, 9, 10, 2, 2, 2],
                                    [1, 0, 2, 1, 0, 0, 0, 3, 4, 5, 1, 1, 6, 7, 2, 2, 2, 8, 9, 10]], dtype=torch.long)
    x_propane_test = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]], dtype=torch.float)

    ei_dep_propane = torch.tensor([[0, 1, 1, 2],
                                   [1, 0, 2, 1]], dtype=torch.long)
    x_dep_propane = torch.tensor([[1], [1], [1]], dtype=torch.float)

    ei_schultz_test = torch.tensor([[0, 1, 1, 2, 1, 3, 3, 4],
                                    [1, 0, 2, 1, 3, 1, 4, 3]], dtype=torch.long)
    x_schultz_test = torch.tensor([[1], [1], [1], [1], [1]], dtype=torch.float)

    data_directed = Data(x=x_directed, edge_index=edge_index_directed)
    data_empty = Data(x=x_empty, edge_index=edge_index_empty)
    data_example = Data(x=x_example, edge_index=edge_index_example)
    data_example_bigger = Data(x=x_example_bigger, edge_index=edge_index_example_bigger)
    data_example_pi = Data(x=x_example_bigger, edge_index=edge_index_example_pi)
    data_balaban_test = Data(x=x_balban_test, edge_index=ei_balban_test)
    data_schultz = Data(x=x_schultz_test, edge_index=ei_schultz_test)
    data_propane = Data(x=x_propane_test, edge_index=ei_propane_test)
    data_depleted_propane = Data(x=x_dep_propane, edge_index=ei_dep_propane)

    def test_num_edges(self):
        self.assertTrue(self.data_directed.num_edges, 3)
        self.assertTrue(self.data_example.num_edges, 2)

    def test_create_zagreb_index(self):
        expected = np.array(6)
        result = create_zagreb_index(self.data_example)
        self.assertTrue(np.array_equal(result, expected), f'calculating zagreb_index failed: expected {expected} but '
                                                          f'got {result}')

    def test_create_zagreb_index_empty_graph(self):
        expected = np.array(0)
        result = create_zagreb_index(self.data_empty)
        self.assertTrue(np.array_equal(result, expected), f'calculating zagreb_index failed: expected {expected} but '
                                                          f'got {result}')

    def test_create_zagreb_index_directed_graph(self):
        self.assertFalse(self.data_directed.is_undirected())
        with self.assertRaises(AssertionError):
            create_zagreb_index(self.data_directed)

    def test_create_polarity_nr(self):
        expected = np.array(0)
        result1 = create_polarity_nr_index(self.data_example)
        self.assertTrue(np.array_equal(result1, expected),
                        f'creating polarity_nr failed: expected {expected} but got {result1}')

    def test_create_polarity_nr_empty(self):
        expected = np.array(0)
        result = create_polarity_nr_index(self.data_empty)
        self.assertTrue(np.array_equal(result, expected),
                        f'creating polarity_nr failed: expected {expected} but got {result}')

    def test_create_polarity_nr_bigger(self):
        expected = np.array(4)
        result = create_polarity_nr_index(self.data_example_bigger)
        self.assertTrue(np.array_equal(result, expected),
                        f'creating polarity_nr failed: expected {expected} but got {result}')

    def test_create_wiener_index(self):
        expected = np.array(4)
        result = create_wiener_index(self.data_example)
        self.assertTrue(np.array_equal(result, expected),
                        f'creating wiener index failed: expected {expected} but got {result}')

    def test_create_wiener_index_empty(self):
        expected = np.array(0)
        result = create_wiener_index(self.data_empty)
        self.assertTrue(np.array_equal(result, expected),
                        f'creating wiener index failed: expected {expected} but got {result}')

    def test_create_wiener_index_bigger(self):
        expected = np.array(31)
        result = create_wiener_index(self.data_example_bigger)
        self.assertTrue(np.array_equal(result, expected),
                        f'creating wiener index failed: expected {expected} but got {result}')

    def test_create_randic_index(self):
        expected = np.array(np.sqrt(2))
        result = create_randic_index(self.data_example)
        self.assertTrue(np.isclose(result, expected),
                        f'creating randic index failed: expected {expected} but got {result}')

    def test_create_estrada_index(self):
        eigenvalues_example = np.array([np.sqrt(2), -np.sqrt(2), 0])
        self.help_test_estrada(eigenvalues_example, self.data_example)

    def test_create_estrada_index_bigger(self):
        eigenvalues_example = np.array([-1.93185, 1.93185, 1, -1, -0.517638, 0.517638])  # eigenvalues calculated with
        # wolfram alpha
        self.help_test_estrada(eigenvalues_example, self.data_example_bigger)

    def help_test_estrada(self, eigenvalues_example, example):
        expected = np.array(0.0)
        for eigenvalue in eigenvalues_example:
            expected += np.exp(eigenvalue)
        result = create_estrada_index(example)
        self.assertTrue(np.isclose(result, expected),
                        f'creating estrada index failed: expected {expected} but got {result}')

    def test_create_balaban_index(self):
        "example from https://de.wikipedia.org/wiki/Balaban-J-Index"
        expected = np.array(3.07437)
        result = create_balaban_index(self.data_balaban_test)
        self.assertTrue(np.isclose(result, expected),
                        f'creating balaban index failed: expected {expected} but got {result}')

    def test_balaban_propane(self):
        "example from https://de.wikipedia.org/wiki/Balaban-J-Index"
        expected = np.array(4.748408577925169)
        g = to_networkx(self.data_propane)
        nx.draw_networkx(g, pos=nx.spring_layout(g), with_labels=False, arrows=False)
        plt.show()
        result = create_balaban_index_wrong(self.data_propane)
        self.assertTrue(np.isclose(result, expected),
                        f'creating balaban index failed: expected {expected} but got {result}')

    def test_dep_prop_balban(self):
        "example from https://de.wikipedia.org/wiki/Balaban-J-Index"
        expected = np.array(3.07437)
        g = to_networkx(self.data_depleted_propane)
        nx.draw_networkx(g, pos=nx.spring_layout(g), with_labels=False, arrows=False)
        plt.show()
        result = create_balaban_index(self.data_depleted_propane)
        self.assertTrue(np.isclose(result, expected),
                        f'creating balaban index failed: expected {expected} but got {result}')

    def test_create_balaban_index_wrong(self):
        "example from https://de.wikipedia.org/wiki/Balaban-J-Index"
        expected = np.array(3.07437)
        result = create_balaban_index_wrong(self.data_balaban_test)
        self.assertTrue(np.isclose(result, expected),
                        f'creating balaban index failed: expected {expected} but got {result}')

    def test_pi_index(self):
        expected = np.array(27)
        result = create_padmakar_ivan_index(self.data_example_pi)
        self.assertTrue(np.isclose(result, expected),
                        f'creating padmakar-ivan index failed: expected {expected} but got {result}')

    def test_szeged_index(self):
        expected = np.array(31)
        result = create_szeged_index(self.data_example_bigger)
        self.assertTrue(np.isclose(result, expected),
                        f'creating szeged index failed: expected {expected} but got {result}')

    def test_schultz_index(self):
        expected = np.array(68)
        result = create_schultz_index(self.data_schultz)
        self.assertTrue(np.isclose(result, expected),
                        f'creating schultz index failed: expected {expected} but got {result}')


if __name__ == '__main__':
    unittest.main()
