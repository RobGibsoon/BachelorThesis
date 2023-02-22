# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
# PTC_MR
dataset = TUDataset(root='tmp/PTC_MR', name='PTC_MR')
print(len(dataset))
print(dataset[0])
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
