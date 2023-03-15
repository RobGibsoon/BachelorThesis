from os.path import exists

import utils
from embedded_graph import EmbeddedGraph
from torch_geometric.datasets import TUDataset
import matplotlib
import pandas as pd

matplotlib.use('TkAgg')


def create_embedded_graph_set(graph_set, wi):
    embedded_graphs = []
    for j in range(len(graph_set)):
        g = EmbeddedGraph(dataset[j], wanted_indices=wi)
        embedded_graphs.append(g)

    return embedded_graphs


def create_dataset(embedded_graphs, wi, dataset_name):
    data = {}
    columns = []
    set_columns_and_df_content(data, columns, wi, embedded_graphs)
    index = []
    for i in range(len(embedded_graphs)):
        index.append(f'graph_{i}')

    create_df_and_save_to_csv(data, columns, index, dataset_name)


def set_columns_and_df_content(data, columns, wi, embedded_graphs):
    wanted_indices = wi
    balaban = []
    nodes = []
    edges = []
    estrada = []
    narumi = []
    padmakar_ivan = []
    polarity_nr = []
    randic = []
    szeged = []
    wiener = []
    zagreb = []
    schultz = []
    for g in embedded_graphs:
        if utils.BALABAN in wanted_indices:
            balaban.append(g.embedding["balaban"])
        if utils.NODES in wanted_indices:
            nodes.append(g.embedding["nodes"])
        if utils.EDGES in wanted_indices:
            edges.append(g.embedding["edges"])
        if utils.ESTRADA in wanted_indices:
            estrada.append(g.embedding["estrada"])
        if utils.NARUMI in wanted_indices:
            narumi.append(g.embedding["narumi"])
        if utils.PADMAKAR_IVAN in wanted_indices:
            padmakar_ivan.append(g.embedding["padmakar_ivan"])
        if utils.POLARITY_NR in wanted_indices:
            polarity_nr.append(g.embedding["polarity_nr"])
        if utils.RANDIC in wanted_indices:
            randic.append(g.embedding["randic"])
        if utils.SZEGED in wanted_indices:
            szeged.append(g.embedding["szeged"])
        if utils.WIENER in wanted_indices:
            wiener.append(g.embedding["wiener"])
        if utils.ZAGREB in wanted_indices:
            zagreb.append(g.embedding["zagreb"])
        if utils.SCHULTZ in wanted_indices:
            schultz.append(g.embedding["schultz"])

    if utils.BALABAN in wanted_indices:
        data["balaban"] = balaban
        columns.append('balaban')
    if utils.NODES in wanted_indices:
        data["nodes"] = nodes
        columns.append('nodes')
    if utils.EDGES in wanted_indices:
        data["edges"] = edges
        columns.append('edges')
    if utils.ESTRADA in wanted_indices:
        data["estrada"] = estrada
        columns.append('estrada')
    if utils.NARUMI in wanted_indices:
        data["narumi"] = narumi
        columns.append('narumi')
    if utils.PADMAKAR_IVAN in wanted_indices:
        data["padmakar_ivan"] = padmakar_ivan
        columns.append('padmakar_ivan')
    if utils.POLARITY_NR in wanted_indices:
        data["polarity_nr"] = polarity_nr
        columns.append('polarity_nr')
    if utils.RANDIC in wanted_indices:
        data["randic"] = randic
        columns.append('randic')
    if utils.SZEGED in wanted_indices:
        data["szeged"] = szeged
        columns.append('szeged')
    if utils.WIENER in wanted_indices:
        data["wiener"] = wiener
        columns.append('wiener')
    if utils.ZAGREB in wanted_indices:
        data["zagreb"] = zagreb
        columns.append('zagreb')
    if utils.SCHULTZ in wanted_indices:
        data["schultz"] = schultz
        columns.append('schultz')


def create_df_and_save_to_csv(data, columns, index, dataset_name):
    df = pd.DataFrame(data, columns=columns, index=index)
    if not exists(f'C:/Users/Robin/BachelorThesis/BachelorThesis/embedded_{dataset_name}.csv'):
        df.to_csv(f'C:/Users/Robin/BachelorThesis/BachelorThesis/embedded_{dataset_name}.csv')
    else:
        i = 0
        while exists(f'C:/Users/Robin/BachelorThesis/BachelorThesis/embedded_{dataset_name}_{i}.csv'):
            i += 1
        df.to_csv(f'C:/Users/Robin/BachelorThesis/BachelorThesis/embedded_{dataset_name}_{i}.csv')


if __name__ == "__main__":
    dataset = TUDataset(root='/tmp/PTC_MR', name='PTC_MR')
    dataset_name = dataset.name
    wanted_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    embedded_graph_set = create_embedded_graph_set(dataset, wanted_indices)
    create_dataset(embedded_graph_set, wanted_indices, dataset_name)
