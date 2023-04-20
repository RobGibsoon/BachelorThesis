from os.path import exists

from utils import BALABAN, ESTRADA, NARUMI, PADMAKAR_IVAN, POLARITY_NR, RANDIC, SZEGED, WIENER, ZAGREB, NODES, EDGES, \
    SCHULTZ, is_connected, log
from embedded_graph import EmbeddedGraph
from torch_geometric.datasets import TUDataset
import matplotlib
import pandas as pd

matplotlib.use('TkAgg')
DIR = "dataset"

def create_embedded_graph_set(dataset, wanted_indices):
    embedded_graphs = []
    successful_count = 0
    unsuccessful_count = 0
    for i in range(len(dataset)):
        if i % 50 == 0:
            log(f'Successfully Embedded {successful_count}/{len(dataset)} graphs', DIR)
            log(f'Failed embedding on {unsuccessful_count}/{len(dataset)} graphs', DIR)
        try:
            g = EmbeddedGraph(dataset[i], wanted_indices=wanted_indices)
            embedded_graphs.append(g)
            successful_count += 1
        except Exception as e:
            log(e, dir)
            assert not is_connected(dataset[i])
            unsuccessful_count += 1
    log(f'Finished embedding with successfully on {successful_count}/{len(dataset)} graphs but failed on '
          f'{unsuccessful_count}/{len(dataset)} graphs', DIR)
    return embedded_graphs


def create_dataset(embedded_graph_set, wanted_indices, dataset_name):
    data = {}
    set_df_content(data, wanted_indices, embedded_graph_set)
    create_df_and_save_to_csv(data, dataset_name)


def set_df_content(data, wanted_indices, embedded_graph_set):
    labels = []
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
    for g in embedded_graph_set:
        labels.append(g.y.item())
        if BALABAN in wanted_indices:
            balaban.append(g.embedding["balaban"])
        if NODES in wanted_indices:
            nodes.append(g.embedding["nodes"])
        if EDGES in wanted_indices:
            edges.append(g.embedding["edges"])
        if ESTRADA in wanted_indices:
            estrada.append(g.embedding["estrada"])
        if NARUMI in wanted_indices:
            narumi.append(g.embedding["narumi"])
        if PADMAKAR_IVAN in wanted_indices:
            padmakar_ivan.append(g.embedding["padmakar_ivan"])
        if POLARITY_NR in wanted_indices:
            polarity_nr.append(g.embedding["polarity_nr"])
        if RANDIC in wanted_indices:
            randic.append(g.embedding["randic"])
        if SZEGED in wanted_indices:
            szeged.append(g.embedding["szeged"])
        if WIENER in wanted_indices:
            wiener.append(g.embedding["wiener"])
        if ZAGREB in wanted_indices:
            zagreb.append(g.embedding["zagreb"])
        if SCHULTZ in wanted_indices:
            schultz.append(g.embedding["schultz"])


    data["labels"] = labels
    if BALABAN in wanted_indices:
        data["balaban"] = balaban
    if NODES in wanted_indices:
        data["nodes"] = nodes
    if EDGES in wanted_indices:
        data["edges"] = edges
    if ESTRADA in wanted_indices:
        data["estrada"] = estrada
    if NARUMI in wanted_indices:
        data["narumi"] = narumi
    if PADMAKAR_IVAN in wanted_indices:
        data["padmakar_ivan"] = padmakar_ivan
    if POLARITY_NR in wanted_indices:
        data["polarity_nr"] = polarity_nr
    if RANDIC in wanted_indices:
        data["randic"] = randic
    if SZEGED in wanted_indices:
        data["szeged"] = szeged
    if WIENER in wanted_indices:
        data["wiener"] = wiener
    if ZAGREB in wanted_indices:
        data["zagreb"] = zagreb
    if SCHULTZ in wanted_indices:
        data["schultz"] = schultz


def create_df_and_save_to_csv(data, dataset_name):
    df = pd.DataFrame(data)
    print(df, DIR)
    if not exists(f'../embedded_{dataset_name}.csv'):
        df.to_csv(f'../embedded_{dataset_name}.csv', index=False)
    else:
        i = 0
        while exists(f'../embedded_{dataset_name}_{i}.csv'):
            i += 1
        df.to_csv(f'../embedded_{dataset_name}_{i}.csv', index=False)


if __name__ == "__main__":
    dataset = TUDataset(root='/tmp/PTC_MR', name='PTC_MR')
    dataset_name = dataset.name
    wanted_indices = [BALABAN, ESTRADA, NARUMI, PADMAKAR_IVAN, POLARITY_NR, RANDIC, SZEGED, WIENER, ZAGREB, NODES,
                      EDGES, SCHULTZ]
    embedded_graph_set = create_embedded_graph_set(dataset, wanted_indices)
    create_dataset(embedded_graph_set, wanted_indices, dataset_name)
