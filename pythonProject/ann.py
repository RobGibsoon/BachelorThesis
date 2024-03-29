from sklearn.model_selection import KFold
from torch import nn
import torch
import numpy as np
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import Dataset, DataLoader
from utils import log

DIR = "embedding_classifier"


def mean_score_ann(X_train, y):
    """calculates the mean score of an ANN clf using cross-validation"""
    clf = ANN(X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    epochs = 200
    batch_size = 1
    k = 5
    splits = KFold(n_splits=k, shuffle=True, random_state=42)
    mean_val_acc = 0
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(X_train)))):
        log('Fold {}'.format(fold + 1), DIR)

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
            log(f'[{epoch + 1}/{epochs}] loss: {running_loss / 2000:.5f}', DIR)

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
