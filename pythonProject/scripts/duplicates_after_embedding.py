import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx

"""this file is to save code that I used to check for duplicates in ptc_mr"""


def check_duplicates(dataset, embedded_graphs):
    """this code was used to show that for dataset PTC_MR duplicate graphs exist when only looking at nodes and edges
        for example graphs 97 and 96"""

    embeddings = np.array([])
    for i in range(len(embedded_graphs)):
        embeddings = np.append(embeddings, list(embedded_graphs[i].embedding.values()))

    embeddings = embeddings.reshape(344,12)

    length = len(np.unique(np.array(embeddings, dtype=float), axis=0))
    arr = np.array(embeddings, dtype=float)
    unique_vals, counts = np.unique(arr, axis=0, return_counts=True)
    a, b= dataset[96], dataset[97]
    g, h = to_networkx(a), to_networkx(b)
    nx.draw_networkx(g, pos=nx.spring_layout(g), with_labels=False, arrows=False)
    plt.show()
    nx.draw_networkx(h, pos=nx.spring_layout(h), with_labels=False, arrows=False)
    plt.show()