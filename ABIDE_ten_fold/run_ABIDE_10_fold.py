import os
import h5py
import numpy as np
from dgl.data import DGLDataset
import matplotlib.pyplot as plt
from time import gmtime, strftime
from model import *
import dgl
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
from tqdm import tqdm

ROI_N = 200
frames = 315


fold = 10
accuracies = []
os.system('mkdir tmp')
os.system('mkdir FC')
for fold_i in range(fold):
    print('--- fold %d:'%fold_i)
    os.system('mkdir tmp/%d'%fold_i)
    ########################################## Load data ########################################
    # Download from: https://drive.google.com/file/d/1RhMRzDRT2vAkXDiW4t55Wbt8XRi6f9_x/view?usp=sharing
    with h5py.File('ABIDE_I_10_fold.h5', 'r') as f:
        x_train, y_train = [], []
        for fold_ii in range(fold):
            if fold_ii == fold_i:
                x_val = f[str(fold_ii)]['X'][()]
                y_val = f[str(fold_ii)]['Y'][()]
            else:
                x_train.append(f[str(fold_ii)]['X'][()])
                y_train.append(f[str(fold_ii)]['Y'][()])
    y_train = np.concatenate(y_train, 0).astype(np.float32)
    y_val = y_val.astype(np.float32)
    
    graph_path = 'FC/FC_no_fold_%d.npy'%fold_i
    if not os.path.exists(graph_path):
        FC = []
        for x_fold in x_train:
            for x in x_fold:
                idx = np.where(x.sum(1) == 0) # find non-zero frames
                if not idx[0].size:
                    tmp = x
                else:
                    tmp = x[:idx[0][0]]
                FC.append(np.corrcoef(tmp.T))
        FC = np.stack(FC, 0)
        FC = np.nan_to_num(FC)
        np.save(graph_path, FC.mean(0))

    x_train = np.expand_dims(np.concatenate(x_train, 0), -1) # (None, 315, 200, 1)
    x_val = np.expand_dims(x_val, -1)
    assert x_train.shape[0] == y_train.shape[0]
    assert x_val.shape[0] == y_val.shape[0]
    assert x_train.shape[0] + x_val.shape[0] == 1057
    print (x_train.shape)
    print (x_val.shape)
    ################################ Set parameter ###############################
    print()
    weight_name = None

    k = 5
    print('k:', k)
    batch_size = 1
    epochs = 50
    l2_reg = 1e-3
    kernels = [8,16,32,64,128]
    lr = 5e-4

    print('kernels:', kernels)
    print('l2:', l2_reg)
    print('batch_size:', batch_size)
    print('epochs:', epochs)
    print('lr:', lr)

    folder = 'tmp/%d/'%(fold_i)

    file_name=folder+'k_%d_l2_%g'%(k, l2_reg)
    print('file_name:', file_name)

    tmp_name = file_name + '_' + strftime("%Y_%m_%d_%H_%M_%S", gmtime()) + '.tmp'
    print('output tmp name:', tmp_name)




    class GCN(nn.Module):
        def __init__(self, in_feats, h_feats, num_classes):
            super(GCN, self).__init__()
            self.conv1 = GraphConv(in_feats, h_feats)
            self.conv15 = GraphConv(h_feats, h_feats)
            self.conv2 = GraphConv(h_feats, num_classes)

        def forward(self, g, in_feat):
            h = self.conv1(g, in_feat)
            h = F.relu(h)
            h = self.conv15(g, h)
            h = F.relu(h)
            h = self.conv2(g, h)
            g.ndata['h'] = h
            out = torch.sigmoid(dgl.mean_nodes(g, 'h'))
            return out


    model = GCN(315, 16, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



    ######################################## Training ####################################################

    print('load graph:', graph_path)
    adj_matrix = np.load(graph_path)
    graph = adj_matrix.argsort(axis=1)[:, ::-1][:, 1:k + 1]
    src = []
    dst = []
    for i in range(len(graph)):
        for j in range(k):
            src.append(i)
            dst.append(graph[i][j])

    x_graphs = []
    for i in range(x_train.shape[0]):
        g = dgl.graph((src, dst))
        g.ndata['feat'] = torch.from_numpy(x_train[i].reshape((200, 315)))
        x_graphs.append(g)


    train = [x for x in zip(x_graphs, torch.from_numpy(y_train))]
    random.shuffle(train)

    print('Train...')
    model.train()
    for epoch in range(12):
        train_loss = 0.0
        accuracy = 0.0
        for graph, label in tqdm(train):
            pred = model(graph, graph.ndata['feat'].float())
            loss = F.binary_cross_entropy(pred.view(-1), label.view(-1))
            if round(pred.item()) == label.item():
                accuracy+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f'Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(x_graphs)} \t\t Training Acc: {accuracy / len(x_graphs)}')


    # ######################################## Val and Test ####################################################

    print('Val and Test...')

    val_graphs = []
    for i in range(x_val.shape[0]):
        g = dgl.graph((src, dst))
        g.ndata['feat'] = torch.from_numpy(x_train[i].reshape((200, 315)))
        val_graphs.append(g)

    val = [x for x in zip(val_graphs, torch.from_numpy(y_val))]
    val_loss = 0.0
    val_accuracy = 0.0
    for graph, label in tqdm(val):
        model.eval()
        pred = model(graph, graph.ndata['feat'].float())
        loss = F.binary_cross_entropy(pred.view(-1), label.view(-1))
        if round(pred.item()) == label.item():
            val_accuracy += 1
        val_loss += loss.item()

    print(
        f'Val Loss: {val_loss / len(val_graphs)} \t\t Training Acc: {val_accuracy / len(val_graphs)}')
    accuracies.append(val_accuracy / len(val_graphs))

print(accuracies)
