import numpy as np
import utils
from torch_geometric.data import Data
from indices import create_zagreb_index, create_polarity_nr_index, \
    create_wiener_index, create_randic_index, create_estrada_index, create_balaban_index, create_padmakar_ivan_index, \
    create_szeged_index, create_narumi_index, create_schultz_index


class EmbeddedGraph(Data):
    """Extends the data in order to add my own embedding for a graph"""

    def __init__(self, data, wanted_indices, **kwargs):
        super().__init__(x=data.x, edge_index=data.edge_index, y=data.y, **kwargs)
        try:
            self.embedding = self.set_embedding(data, wanted_indices)
        except:
            raise Exception("Something went wrong when trying to get the embedding for this graph: ",
                            Exception.__name__)

    def set_embedding(self, graph, wanted_indices):
        """wanted_indices is a list with the names of which indices we want in the embedding"""
        embedding = {}
        if utils.BALABAN in wanted_indices:
            embedding["balaban"] = create_balaban_index(graph)
        if utils.NODES in wanted_indices:
            embedding["nodes"] = np.array([graph.num_nodes])
        if utils.EDGES in wanted_indices:
            embedding["edges"] = np.array([int(len(graph.edge_index[1]) / 2)])
        if utils.ESTRADA in wanted_indices:
            embedding["estrada"] = create_estrada_index(graph)
        if utils.NARUMI in wanted_indices:
            embedding["narumi"] = create_narumi_index(graph)
        if utils.PADMAKAR_IVAN in wanted_indices:
            embedding["padmakar_ivan"] = create_padmakar_ivan_index(graph)
        if utils.POLARITY_NR in wanted_indices:
            embedding["polarity_nr"] = create_polarity_nr_index(graph)
        if utils.RANDIC in wanted_indices:
            embedding["randic"] = create_randic_index(graph)
        if utils.SZEGED in wanted_indices:
            embedding["szeged"] = create_szeged_index(graph)
        if utils.WIENER in wanted_indices:
            embedding["wiener"] = create_wiener_index(graph)
        if utils.ZAGREB in wanted_indices:
            embedding["zagreb"] = create_zagreb_index(graph)
        if utils.SCHULTZ in wanted_indices:
            embedding["schultz"] = create_schultz_index(graph)

        return embedding
