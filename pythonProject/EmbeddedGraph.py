from GraphUtils import get_all_indices
from torch_geometric.data import Data


class EmbeddedGraph(Data):

    def __init__(self, data, **kwargs):
        super().__init__(x=data.x, edge_index=data.edge_index, y=data.y, **kwargs)
        self.embedding = get_all_indices(data)
