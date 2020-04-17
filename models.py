


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import time
from datetime import datetime

import pdb
import numpy as np
import torch
import torch.optim as optim

import torch_geometric.transforms as T
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt



class GNNmsa(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GNNmsa, self).__init__()
        self.task = task
        # self.convs = nn.ModuleList()
        self.conv1 = self.build_conv_model(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        # self.conv2 = self.build_conv_model(hidden_dim, hidden_dim)
        # self.ln2 = nn.LayerNorm(hidden_dim)

        # self.conv3 = self.build_conv_model(hidden_dim, 16)
        # self.ln3 = nn.LayerNorm(hidden_dim)

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 2

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
          x = torch.ones(data.num_nodes, 1)
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ln1(x)

        # x = self.conv2(x, edge_index)
        emb = x
        # x = F.relu(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.ln2(x)

        # x = self.conv3(x, edge_index)
        # emb = x
        # x = F.relu(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(1):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.3
        self.num_layers = 2

    def build_conv_model(self, input_dim, hidden_dim):
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
          x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)
        x = self.post_mp(x)
        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Net, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = pyg_nn.GCNConv(hidden_channels, hidden_channels)
        self.fc1 = nn.Linear(hidden_channels, 1)
        self.sig = nn.Sigmoid()
        # Saeed did up to here

    def forward(self, data):
        x, edge_index = data.x, data.edge_index 
        x = nn.functional.tanh(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        emb = x
        x = self.sig(self.fc1(x))
        return emb, x

    def loss(self, pred, label):
        criterion = nn.MSELoss()
        loss = criterion(pred.squeeze(), label.reshape(len(label),1).float())
        return loss


def test_influence():
    data = KarateClub()[0]
    x = torch.randn(data.num_nodes, 8)

    out = influence(Net(x.size(1), 16), x, data.edge_index)
    assert out.size() == (data.num_nodes, data.num_nodes)
    assert torch.allclose(
        out.sum(dim=-1), torch.ones(data.num_nodes), atol=1e-04)


