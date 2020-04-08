
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import time
from datetime import datetime

import networkx as nx
import pdb
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import KarateClub
from torch_geometric.data import DataLoader

import torch_geometric.transforms as T

from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class GNNmsa(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GNNmsa, self).__init__()
        self.task = task
        # self.convs = nn.ModuleList()
        h1s = 32 
        self.conv1 = self.build_conv_model(input_dim, h1s)
        self.ln1 = nn.LayerNorm(h1s)

        self.conv2 = self.build_conv_model(h1s, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.conv3 = self.build_conv_model(hidden_dim, 32)
        self.ln3 = nn.LayerNorm(hidden_dim)

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 3

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

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ln2(x)

        x = self.conv3(x, edge_index)
        emb = x
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

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
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 3

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

        # TODO: only for karate:
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


def train(dataset, task, writer):
    if task == 'graph':
        data_size = len(dataset)
        loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=True)
    else:
        print("loading dataloader")
        test_loader = loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # build model
    if task == "karate":
        model = GNNStack(1, 32, dataset.num_classes, task="node")
    else:
        model = GNNStack(max(dataset.num_node_features, 1), 32, dataset.num_classes, task=task)
    # model = GNNmsa(max(dataset.num_node_features, 1), 32, dataset.num_classes, task=task)

    # opt = optim.Adam(model.parameters(), lr=0.01)
    opt = optim.SGD(model.parameters(), lr=0.01)
    MILESTONE=[50, 100, 200, 400, 500]
    train_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=MILESTONE, gamma=0.2)

    # train
    for epoch in range(100):
        total_loss = 0
        total_count = 0
        total_correct = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
            if task == 'node':
                # notice the mask --
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
            total_count += len(pred)
            total_correct += sum(torch.argmax(pred, 1) == label)
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
            # print("This batch: " + str(len(pred)))
        total_loss /= len(loader.dataset)
        train_acc = total_correct / float(total_count)
        writer.add_scalar("loss", total_loss, epoch)

        if epoch % 5 == 0:
            test_acc = test(test_loader, model)
            print("Epoch {}. \tLoss: {:.4f} \tTrain acc: {:.4f} \tTest acc: {:.4f}".format(
                epoch, total_loss, train_acc, test_acc))
            writer.add_scalar("test accuracy", test_acc, epoch)

    return model



def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            emb, pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y
        
        # Saeed issue with mask in karate dataset
        #if model.task == 'node':
        #     mask = data.val_mask if is_validation else data.test_mask
        #     # node classification: only evaluate on nodes in test set
        #     pred = pred[mask]
        #     label = data.y[mask]
            
        correct += pred.eq(label).sum().item()
    
    if model.task == 'graph':
        total = len(loader.dataset) 
    else:
        total = 0
        for data in loader.dataset:
            # total += torch.sum(data.test_mask).item()
            total += len(data.y)
    return correct / total


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


if __name__ == "__main__":
    
    task = "karate"

    if task == "graph":
        writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        dataset = dataset.shuffle()
        model = train(dataset, task, writer)

    elif task == "node":
        # Planetoid: Cora, CiteSeer, PubMed
        ds = "Cora"
        writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        dataset = Planetoid(root='/tmp/'+ds, name=ds)
        # set_dataset_masks(dataset, 0.4, 0.2)
        model = train(dataset, task, writer)
        # writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        # dataset = CitationFull(root='/tmp/citeseer', name='citeseer')
        # model = train(dataset, task, writer)

    elif task == "karate":
        writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        dataset = KarateClub()
        pdb.set_trace()
        model = train(dataset, task, writer)


