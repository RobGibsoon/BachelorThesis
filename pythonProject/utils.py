from itertools import chain
import pandas as pd

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch import combinations, nn
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import to_dense_adj, degree
from scipy.sparse.csgraph import dijkstra

feature_names = {0: "balaban", 1: "estrada", 2: "narumi", 3: "padmakar-ivan", 4: "polarity-nr", 5: "randic",
                 6: "szeged", 7: "wiener", 8: "zagreb", 9: "nodes", 10: "edges", 11: "schultz"}
BALABAN = 0
ESTRADA = 1
NARUMI = 2
PADMAKAR_IVAN = 3
POLARITY_NR = 4
RANDIC = 5
SZEGED = 6
WIENER = 7
ZAGREB = 8
NODES = 9
EDGES = 10
SCHULTZ = 11

NP_SEED = 42

np.random.seed(NP_SEED)


def get_degrees(graph):
    edge_index = graph.edge_index[0]
    num_nodes = graph.num_nodes
    return degree(edge_index, num_nodes)


def get_distance_matrix(graph):
    """for a graph returns a np-array (num_nodes, num_nodes) with the shortest path distances for each of the nodes"""
    assert graph.is_undirected()
    adj = to_dense_adj(graph.edge_index)
    graph = np.squeeze(adj.numpy(), axis=0)  # collapse outer dimension to get (n,n) array
    graph = graph.tolist()
    if len(graph) == 0:
        return np.array([0])
    csr_mat = csr_matrix(graph)
    dist_matrix = dijkstra(csgraph=csr_mat, directed=False)
    return dist_matrix


def all_subsets(indices_list):
    """returns all possible combinations of sets of indices"""
    return chain(*map(lambda x: combinations(torch.from_numpy(indices_list), x), range(0, len(indices_list) + 1)))


def best_subset_ann(X_train, X_test, y):
    """returns the best set for classifying using an ANN clf, uses cross-validation and takes the set with the highest
    mean accuracy"""
    n_features = X_train.shape[1]
    best_score = -np.inf
    best_subset = None
    count = 1

    for subset in all_subsets(np.arange(n_features)):
        score = mean_score_ann(np.reshape(X_train[:, subset], (X_train.shape[0], -1)),
                               np.reshape(X_test[:, subset], (X_test.shape[0], -1)),
                               y)
        if score > best_score:
            best_score, best_subset = score, subset
        print(f'subset {count}/{2 ** n_features - 1}')
        count += 1

    print(f"best_subset: {best_subset.numpy()} with best score: {best_score}")
    return best_subset.numpy(), best_score


def mean_score_ann(X_train, X_test, y):
    """calculates the mean score of an ANN clf using cross-validation"""
    # question: is this implementation of cv correct? it is correct to not use the X_test right?
    clf = ANN(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    epochs = 200
    batch_size = 1
    k = 5
    splits = KFold(n_splits=k, shuffle=True, random_state=42)
    mean_val_acc = 0
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(X_train)))):
        print('Fold {}'.format(fold + 1))

        train_data = Data(X_train[train_idx, :], y[train_idx])
        test_data = Data(X_train[val_idx, :], y[val_idx])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_data, batch_size=batch_size)
        val_acc = train_ann(clf, epochs, criterion, train_loader, test_loader)[0]
        mean_val_acc += val_acc

    mean_val_acc /= k
    return mean_val_acc


def train_ann(clf, epochs, criterion, train_loader, test_loader):
    """used to train an ANN"""
    optimizer = torch.optim.SGD(clf.parameters(), lr=0.1)
    scheduler = CyclicLR(optimizer,
                         base_lr=0.0001,
                         # Initial learning rate which is the lower boundary in the cycle for each parameter group
                         max_lr=1e-3,  # Upper learning rate boundaries in the cycle for each parameter group
                         step_size_up=4,  # Number of training iterations in the increasing half of a cycle
                         mode="triangular")
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = clf(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f'[{epoch + 1}/{epochs}] loss: {running_loss / 2000:.5f}')

    correct, total = 0, 0
    predictions = []
    labels_total = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            # calculate output by running through the network
            outputs = clf(inputs)
            # get the predictions
            __, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.item())
            labels_total.append(labels.item())
            # update results
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total, predictions, labels


def best_subset_svm_knn(estimator, X_train, y, cv=3):
    """returns the best set for classifying using an SVM/KNN clf, uses cross-validation and takes the set with the
    highest mean accuracy"""
    n_features = X_train.shape[1]
    best_score = -np.inf
    best_subset = None
    count = 1

    for subset in all_subsets(np.arange(n_features)):
        score = cross_val_score(estimator, np.reshape(X_train[:, subset], (X_train.shape[0], -1)), y, cv=cv).mean()
        if score > best_score:
            best_score, best_subset = score, subset
        print(f'subset {count}/{2 ** n_features - 1}')
        count += 1

    print(f"best_subset: {best_subset.numpy()} with best score: {best_score}")
    return best_subset.numpy(), best_score


def feature_selected_sets(clf, X_train, X_test, y):
    """returns the modified training and test sets after performing feature selection on them"""
    if not isinstance(clf, (KNeighborsClassifier, SVC)):
        best_subset, best_score = best_subset_ann(X_train, X_test, y)
    else:
        best_subset, best_score = best_subset_svm_knn(clf, X_train, y)
    features = get_feature_names(best_subset)
    print(f"The optimal features selected for {type(clf).__name__} were: {features}")
    X_train_fs = X_train[:, best_subset]
    X_test_fs = X_test[:, best_subset]
    assert (X_train_fs.shape[0], len(best_subset)) == X_train_fs.shape
    return X_train_fs, X_test_fs


def get_feature_names(feature_subset):
    """returns the names of the features for a list with indices 0-11"""
    assert len(feature_subset) > 0
    features = ''
    for i, feature in enumerate(feature_subset):
        if i + 1 != len(feature_subset):
            features += (feature_names[feature])
            features += ", "
        else:
            features += (feature_names[feature])
    return features


def save_preds(preds, labels, clf, dataset_name, feature_selection):
    """saves labels and predictions to a csv-file"""
    data = {"preds": preds, "labels": labels}
    df = pd.DataFrame(data)
    df.to_csv(f'../preds_labels_{clf}_{dataset_name}_fs{feature_selection}.csv', index=False)


class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()
        self.input_dim = 12
        output_dim = 2
        hidden_layers = (input_dim + output_dim) // 2  # the mean between input_dim + output_dim
        self.linear1 = nn.Linear(input_dim, hidden_layers)
        self.linear2 = nn.Linear(hidden_layers, output_dim)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x


class Data(Dataset):
    def __init__(self, X_train, y_train):
        self.X = torch.from_numpy(X_train.astype(np.float32))
        self.y = torch.from_numpy(y_train).type(torch.LongTensor)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len
