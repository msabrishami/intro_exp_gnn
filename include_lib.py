

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

#ds = "Cora"
#writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
#dataset = Planetoid(root='/tmp/'+ds, name=ds)

writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
dataset = KarateClub()

