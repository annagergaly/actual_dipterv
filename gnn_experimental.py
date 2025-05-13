import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
import pytorch_dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from itertools import islice
from torch_geometric.datasets import NeuroGraphDataset

# dataset = pytorch_dataset.AutismDataset(['NYU'], node_features='average')
dataset = NeuroGraphDataset(root='pyG', name='HCPGender')

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 32)
        torch.nn.init.xavier_uniform(self.conv1.lin.weight)
        self.conv2 = GCNConv(32, 16)
        torch.nn.init.xavier_uniform(self.conv2.lin.weight)
        self.fully_connected = nn.Linear(16*1000, 64)#dataset.num_classes)
        torch.nn.init.xavier_uniform(self.fully_connected.weight)
        self.fully_connected2 = nn.Linear(64, 1)
        torch.nn.init.xavier_uniform(self.fully_connected2.weight)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        #print("orig", x, x.shape)
        #print(edge_weight, edge_weight.shape)
        x = self.conv1(x, edge_index, edge_weight)
        ##print("conv1", x, x.shape)
        x = F.relu(x)
        #print("conv1 relu", x.count_nonzero())
        #print(torch.isnan(x.view(-1)).sum())
        x = self.conv2(x, edge_index, edge_weight)
        ##print("conv2", x, x.shape)
        x = F.relu(x)
        #print("conv2 relu", x.count_nonzero())
        #print(torch.isnan(x.view(-1)).sum())
        x = x.view(-1, 16*1000)
        ## print("view",x, x.shape)
        x = self.fully_connected(x)
        x= F.relu(x)
        x = self.fully_connected2(x)
        #print("fc", x, x.shape)

        return F.sigmoid(x)#F.softmax(x, dim=1)
    
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device", device)

model = GCN().to(device)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_dataloader = DataLoader(train_dataset, 1, True)
test_dataloader = DataLoader(test_dataset, 1, True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)


model.train()
for epoch in tqdm(range(1)):
    for data in islice(train_dataloader, 8):
        data = data.to(device)
        # print(data.edge_index)
        optimizer.zero_grad()
        out = model(data).view(-1, 1)
        #loss = F.nll_loss(out, data.y)
        #print("out", out, out.shape, out.dtype)
        #print("y", data.y, data.y.shape, data.y.dtype)
        loss = F.binary_cross_entropy(out, data.y.view(-1, 1).float())
        print("loss", loss)
        loss.backward()
        optimizer.step()

correct = 0
model.eval()
for data in test_dataloader:
    data = data.to(device)
    # print(data.x, data.x.shape)
    # print(data.edge_index, data.edge_index.shape)
    # print(data.edge_attr, data.edge_attr.shape)
    # print("model(data)", model(data), model(data).shape, model(data).dtype)
    pred = torch.round(model(data)).to(dtype=torch.int64).squeeze()
    print("pred", pred, data.y)
    print(pred == data.y)
    correct += (pred == data.y).sum()
    print("correct", correct)

#for data in test_dataloader:
#    print(data.y)

# correct = 0
# model.eval()
# for data in test_dataloader:
#     data = data.to(device)
#     print("model(data)", model(data), model(data).shape, model(data).dtype)
#     pred = torch.round(model(data)).view(-1, 1)
#     #pred = model(data).argmax(dim=1)
#     print(pred)
#     correct += (pred == data.y).sum()
    
acc = int(correct) / int(test_dataloader.dataset.__len__())
print(f'Accuracy: {acc:.4f}')
print(device)

# print("conv1.weight", model.conv1.lin.weight)
# print("conv2.weight", model.conv2.lin.weight)
# print("fc.weight", model.fully_connected.weight)
# print("fc2.weight", model.fully_connected2.weight)