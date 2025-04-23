import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
import pytorch_dataset
from torch_geometric.loader import DataLoader

dataset = pytorch_dataset.AutismDataset(['NYU'])

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.fully_connected = nn.Linear(16*dataset.num_nodes, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = x.view(-1, 16*dataset.num_nodes)
        x = self.fully_connected(x)

        return F.log_softmax(x, dim=1)
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_dataloader = DataLoader(train_dataset, 1, True)
test_dataloader = DataLoader(test_dataset, 1, True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


model.train()
for epoch in range(20):
    for data in train_dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

correct = 0
model.eval()
for data in test_dataloader:
    data = data.to(device)
    pred = model(data).argmax(dim=1)
    correct += (pred == data.y).sum()
    
acc = int(correct) / int(test_dataloader.dataset.__len__())
print(f'Accuracy: {acc:.4f}')
print(device)