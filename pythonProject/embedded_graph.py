import numpy as np
from utils import BALABAN, ESTRADA, NARUMI, PADMAKAR_IVAN, POLARITY_NR, RANDIC, SZEGED, WIENER, ZAGREB, NODES, EDGES, \
    SCHULTZ
from torch_geometric.data import Data
from indices import create_zagreb_index, create_polarity_nr_index, \
    create_wiener_index, create_randic_index, create_estrada_index, create_balaban_index, create_padmakar_ivan_index, \
    create_szeged_index, create_narumi_index, create_schultz_index


class EmbeddedGraph(Data):
    """Extends the data in order to add my own embedding for a graph"""

    def __init__(self, data, wanted_indices, **kwargs):
        super().__init__(x=data.x, edge_index=data.edge_index, y=data.y, **kwargs)
        self.embedding = self.set_embedding(data, wanted_indices)
        self.embedding = self.set_embedding(data, wanted_indices)

    def set_embedding(self, graph, wanted_indices):
        """wanted_indices is a list with the names of which indices we want in the embedding"""
        embedding = {}
        if BALABAN in wanted_indices:
            embedding["balaban"] = create_balaban_index(graph)
        if NODES in wanted_indices:
            embedding["nodes"] = np.array(graph.num_nodes)
        if EDGES in wanted_indices:
            embedding["edges"] = np.array(int(len(graph.edge_index[1]) / 2))
        if ESTRADA in wanted_indices:
            embedding["estrada"] = create_estrada_index(graph)
        if NARUMI in wanted_indices:
            embedding["narumi"] = create_narumi_index(graph)
        if PADMAKAR_IVAN in wanted_indices:
            embedding["padmakar_ivan"] = create_padmakar_ivan_index(graph)
        if POLARITY_NR in wanted_indices:
            embedding["polarity_nr"] = create_polarity_nr_index(graph)
        if RANDIC in wanted_indices:
            embedding["randic"] = create_randic_index(graph)
        if SZEGED in wanted_indices:
            embedding["szeged"] = create_szeged_index(graph)
        if WIENER in wanted_indices:
            embedding["wiener"] = create_wiener_index(graph)
        if ZAGREB in wanted_indices:
            embedding["zagreb"] = create_zagreb_index(graph)
        if SCHULTZ in wanted_indices:
            embedding["schultz"] = create_schultz_index(graph)

        return embedding
