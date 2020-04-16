
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
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Reddit
from aifb import AIFB_pyg

from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from karate_mod import SaeedClub
from models import Net, GNNStack



def set_dataset_masks(ds, train_ratio, test_ratio):
    ''' the right way to do this is by tensors
    and not using for loop, needs to be fixed ''' 
    sz = ds.data.num_nodes
    ds.data.train_mask = torch.tensor([False] * sz)
    ds.data.test_mask = torch.tensor([False] * sz)
    idx = np.random.permutation(dataset.data.num_nodes)
    ds.data.train_mask[idx[:int(train_ratio*sz)]] = True
    ds.data.test_mask[idx[int(train_ratio*sz):int((train_ratio+test_ratio)*sz)]] = True
    ds.slices["train_mask"] = torch.tensor((0, sz))
    ds.slices["test_mask"] = torch.tensor((0, sz))
    if train_ratio + test_ratio != 1:
        ds.data.valid_mask = torch.tensor([False] * sz)
        ds.data.valid_mask[int((train_ratio + test_ratio)*sz):] = True
        ds.slices["valid_mask"] = torch.tensor((0, sz))
    ds.data.is_directed = "salam"
    ds.slices["is_directed"] = 1

def train(dataset, task, writer):
    if task == 'graph':
        data_size = len(dataset)
        loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=True)
    else:
        print("loading dataloader")
        test_loader = loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = GNNStack(max(dataset.num_node_features, 1), 16, dataset.num_classes, task=task)
    # model = GNNmsa(max(dataset.num_node_features, 1), 32, dataset.num_classes, task=task)

    opt = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    # opt = optim.SGD(model.parameters(), lr=0.01)
    # MILESTONE=[50, 100, 200, 400, 500]
    # train_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=MILESTONE, gamma=0.2)
    
    # train
    for epoch in range(400):
        total_loss = 0
        total_count = 0
        total_correct = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
            if task == 'node':
                # pred = pred[batch.train_mask]
                # label = label[batch.train_mask]
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

        if epoch % 2 == 0:
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
        
        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = data.y[mask]
            
        correct += pred.eq(label).sum().item()
    
    if model.task == 'graph':
        total = len(loader.dataset) 
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
    return correct / total


def karate_train(dataset):

    test_loader = loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = Net(max(dataset.num_node_features, 1), 10)
    opt = optim.Adam(model.parameters(), lr=0.01)
    # opt = optim.SGD(model.parameters(), lr=0.1)
    # MILESTONE=[50, 100, 200, 400, 500]
    # train_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=MILESTONE, gamma=0.2)
    criterion = nn.MSELoss()

    for epoch in range(200):
        total_loss = 0
        total_count = 0
        total_correct = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y[batch.train_mask]
            pred = pred[batch.train_mask]
            total_count += len(pred)
            total_correct += sum(torch.argmax(pred, 1) == label)
            # pdb.set_trace()
            # loss = criterion(pred.squeeze(), label.reshape(len(label),1).float())
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
            # print("This batch: " + str(len(pred)))
        total_loss /= len(loader.dataset)
        train_acc = total_correct / float(total_count)
        writer.add_scalar("loss", total_loss, epoch)

        # Testing:
        model.eval()
        correct = 0
        total = 0.0
        for data in test_loader:
            with torch.no_grad():
                emb, pred = model(data)
                pred = pred.argmax(dim=1)
                label = data.y
            
            pred = pred[data.test_mask]
            label = data.y[data.test_mask]    
            correct += pred.eq(label).sum().item()
            total += torch.sum(data.test_mask).item()
        test_acc = correct / total
        print("Epoch {}. \tLoss: {:.4f} \tTrain acc: {:.4f} \tTest acc: {:.4f}".format(
            epoch, total_loss, train_acc, test_acc))
    return model



if __name__ == "__main__":
        
    task = "node"

    if task == "graph":
        writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        dataset = dataset.shuffle()
        model = train(dataset, task, writer)
    
    else:
        writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        # dataset = SaeedClub()
        # dataset = Coauthor(root='/tmp/Coauthor', name="CS")
        # dataset = Amazon(root='/tmp/Amazon', name="Photo")
        # dataset = Reddit(root="/tmp/Reddit")
        # dataset = CitationFull("/tmp/CitationFull", name="cora")
        # dataset = Planetoid(root="/tmp/Cora", name="Cora")
        dataset = AIFB_pyg()
        # set_dataset_masks(dataset, 0.01, 0.2)
        model = train(dataset, task, writer)
        # karate_train(dataset)

'''
    elif task == "node":
        # Planetoid: Cora, CiteSeer, PubMed
        ds = "Cora"
        writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

        # set_dataset_masks(dataset, 0.4, 0.2)
        model = train(dataset, task, writer)
        # writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        # dataset = CitationFull(root='/tmp/citeseer', name='citeseer')
        # model = train(dataset, task, writer)
    '''

