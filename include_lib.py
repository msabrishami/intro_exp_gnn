

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import time
from datetime import datetime

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.datasets import KarateClub

import torch_geometric.transforms as T

from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pdb
from itertools import repeat, product
from karate_mod import SaeedClub 
from aifb import AIFB_pyg

def set_dataset_masks(ds, train_ratio, test_ratio):
    ''' the right way to do this is by tensors
    and not using for loop, needs to be fixed ''' 
    sz = ds[0].num_nodes
    train_sz = int(sz*train_ratio)
    test_sz = int(sz*test_ratio)
    val_sz = sz - (train_sz + test_sz)
    for i in range(sz):
        if i < train_sz:
            ds[0].train_mask[i] = True
            ds[0].test_mask[i] = False
            ds[0].val_mask[i] = False
        elif i < train_sz + test_sz:
            ds[0].train_mask[i] = False
            ds[0].test_mask[i] = True
            ds[0].val_mask[0] = False
        else:
            ds[0].train_mask[i] = False
            ds[0].test_mask[i] = False
            ds[0].val_mask[i] = True

    # ds[0].train_mask = torch.tensor([True]*train_sz + [False]*(sz-train_sz))
    # ds[0].train_mask = torch.tensor([False]*train_sz + [True]*test_sz + [False]*val_sz)
    # ds[0].val_mask = torch.tensor([False]*(train_sz + test_sz) + [True]*val_sz)

# writer2 = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
# ds2 = Planetoid(root='/tmp/Cora', name="Cora")
# writer1 = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
# ds1 = KarateClub()
# 
# print(type(ds1))
# print(type(ds2))
# print(type(ds1.data))
# print(type(ds2.data))
# print(ds1.num_features)
# print(ds2.num_features)
# print(ds1.num_node_features)
# print(ds2.num_node_features)
# print(ds1.data)
# print(ds2.data)
# print(ds1.data.num_nodes)
# print(ds2.data.num_nodes)
# print(type(ds1.data.num_nodes))
# print(type(ds2.data.num_nodes))


ds = AIFB_pyg("/home/msabrishami/.dgl/aifb")


# ds1.data.train_mask = torch.tensor([True] * 34)
# ds3 = SaeedClub()
# 
# print(type(ds1.data.train_mask))
# print(type(ds2.data.train_mask))
# 
# loader1 = DataLoader(ds1, batch_size=4, shuffle=True)
# loader2 = DataLoader(ds2, batch_size=4, shuffle=True)
# loader3 = DataLoader(ds3, batch_size=4, shuffle=True)
# batch1 = next(iter(loader1))
# batch2 = next(iter(loader2))
# print("========================")
# ds = ds2
# for key in ds.data.keys: 
#     print(key)
#     item, slices = ds.data[key], ds.slices[key]
#     if torch.is_tensor(item): 
#         s = list(repeat(slice(None), item.dim()))
# 
