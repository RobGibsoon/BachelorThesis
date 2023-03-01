from torch_geometric.data import Data


class Graph(Data):

    def __init__(self, data):
        self.embedding = None
        print("created graph")
