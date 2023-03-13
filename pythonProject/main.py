# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from embedded_graph import EmbeddedGraph
from torch_geometric.datasets import TUDataset
import matplotlib

matplotlib.use('TkAgg')


def create_embedded_graph_set(graph_set, wi):
    embedded_graphs = []
    for j in range(len(graph_set)):
        g = EmbeddedGraph(dataset[j], wanted_indices=wi)
        # print(g.embedding)
        # g = to_networkx(embedded_graphs[i])
        # nx.draw_networkx(g, pos=nx.spring_layout(g), with_labels=False, arrows=False)
        # plt.show()
        embedded_graphs.append(g)

    return embedded_graphs


if __name__ == "__main__":
    dataset = TUDataset(root='/tmp/PTC_MR', name='PTC_MR')
    wanted_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    embedded_graph_set = create_embedded_graph_set(dataset, wanted_indices)
    for i in [0, 1, 14, 20]:
        print(embedded_graph_set[i].embedding)
