import numpy as np
from torch_geometric.data import Data

from indices import zagreb_index, polarity_nr_index, \
    wiener_index, randic_index, estrada_index, balaban_index, padmakar_ivan_index, \
    szeged_index, narumi_index, schultz_index, avg_edge_stength, modified_zagreb_index, hyper_wiener_index, \
    label_entropy, neighborhood_impurity
from utils import BALABAN, ESTRADA, NARUMI, PADMAKAR_IVAN, POLARITY_NR, RANDIC, SZEGED, WIENER, ZAGREB, NODES, EDGES, \
    SCHULTZ, MOD_ZAGREB, HYP_WIENER, N_IMPURITY, LABEL_ENTROPY, EDGE_STRENGTH


class EmbeddedGraph(Data):
    """Extends the data in order to add my own embedding for a graph"""

    def __init__(self, data, wanted_indices, dataset_name, **kwargs):
        super().__init__(x=data.x, edge_index=data.edge_index, y=data.y, **kwargs)
        self.embedding = self.set_embedding(data, wanted_indices)
        self.dataset_name = dataset_name

    def set_embedding(self, graph, wanted_indices):
        """wanted_indices is a list with the names of which indices we want in the embedding"""
        embedding = {}
        if BALABAN in wanted_indices:
            embedding["balaban"] = balaban_index(graph)
        if NODES in wanted_indices:
            embedding["nodes"] = np.array(graph.num_nodes)
        if EDGES in wanted_indices:
            embedding["edges"] = np.array(int(len(graph.edge_index[1]) / 2))
        if ESTRADA in wanted_indices:
            embedding["estrada"] = estrada_index(graph)
        if NARUMI in wanted_indices:
            embedding["narumi"] = narumi_index(graph)
        if PADMAKAR_IVAN in wanted_indices:
            embedding["padmakar_ivan"] = padmakar_ivan_index(graph)
        if POLARITY_NR in wanted_indices:
            embedding["polarity_nr"] = polarity_nr_index(graph)
        if RANDIC in wanted_indices:
            embedding["randic"] = randic_index(graph)
        if SZEGED in wanted_indices:
            embedding["szeged"] = szeged_index(graph)
        if WIENER in wanted_indices:
            embedding["wiener"] = wiener_index(graph)
        if ZAGREB in wanted_indices:
            embedding["zagreb"] = zagreb_index(graph)
        if SCHULTZ in wanted_indices:
            embedding["schultz"] = schultz_index(graph)
        if MOD_ZAGREB in wanted_indices:
            embedding["mod_zagreb"] = modified_zagreb_index(graph)
        if HYP_WIENER in wanted_indices:
            embedding["hyp_wiener"] = hyper_wiener_index(graph)
        if N_IMPURITY in wanted_indices:
            embedding["n_impurity"] = label_entropy(graph)
        if LABEL_ENTROPY in wanted_indices:
            embedding["label_entropy"] = neighborhood_impurity(graph)
        if EDGE_STRENGTH in wanted_indices:
            embedding["edge_strength"] = avg_edge_stength(graph)

        return embedding
