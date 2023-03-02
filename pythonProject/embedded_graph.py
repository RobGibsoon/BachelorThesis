from indices import get_all_indices
from torch_geometric.data import Data


class EmbeddedGraph(Data):
    """Extends the data in order to add my own embedding for a graph"""

    def __init__(self, data, **kwargs):
        super().__init__(x=data.x, edge_index=data.edge_index, y=data.y, **kwargs)
        try:
            self.embedding = get_all_indices(data)
        except: raise Exception("Something went wrong when trying to get the embedding for this graph: ", Exception.__name__)
