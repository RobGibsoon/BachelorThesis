import argparse
import csv

import numpy as np
import torch
from torch import nn
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from utils import get_csv_idx_split


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # Pooling
        x = global_mean_pool(x, batch)

        # Classifier
        x = self.classifier(x)

        return x


def train_model(model, epochs, criterion, optimizer, scheduler, train_loader, test_loader, device, run, dataset_name):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}')

    model.eval()
    correct = 0
    predictions = []
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        predictions.extend(pred.cpu().numpy())

    with open(f'log/predictions/predictions_gnn_{dataset_name}_{run}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['predictions'])
        writer.writerows(zip(predictions))

    return correct / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dn', type=str,
                        help='Options: PTC_MR, Mutagenicity, MUTAG')
    args = parser.parse_args()
    if args.dn is None:
        raise argparse.ArgumentError(None, "Please pass an index from 0-28.")

    dataset_name = args.dn
    print(dataset_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = TUDataset(root=f'/tmp/{dataset_name}', name=f'{dataset_name}')
    input_dim = data.num_node_features
    hidden_dim = 64
    output_dim = data.num_classes
    epochs = 100
    filter_split = get_csv_idx_split(dataset_name, "filter")
    X = [data[idx] for idx in filter_split]
    train_split = get_csv_idx_split(dataset_name, "train")
    test_split = get_csv_idx_split(dataset_name, "test")
    train_graphs = [X[idx] for idx in train_split]
    test_graphs = [X[idx] for idx in test_split]
    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

    accuracies = []
    for i in range(10):
        model = GNN(input_dim, hidden_dim, output_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.001)
        accuracy = train_model(model, epochs, criterion, optimizer, scheduler, train_loader, test_loader, device, i + 1,
                               dataset_name)
        accuracies.append(accuracy)
        print(f'Run {i + 1}, Accuracy: {accuracy * 100:.2f}%')

    print(f'Average Accuracy over 10 runs: {np.mean(accuracies) * 100:.2f}%')


if __name__ == "__main__":
    main()
