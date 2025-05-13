import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric
import torch_geometric.nn
from tqdm import tqdm
import pytorch_dataset
from torch_geometric.loader import DataLoader

NUM_NODES = 111

class GCNModel(nn.Module):
    def __init__(self, num_node_features):
        super(GCNModel, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(num_node_features, 32)
        # torch.nn.init.xavier_uniform_(self.conv1.lin.weight)
        self.conv2 = torch_geometric.nn.GCNConv(32, 16)
        # torch.nn.init.xavier_uniform_(self.conv2.lin.weight)
        self.fully_connected = nn.Linear(16*NUM_NODES, 64)
        # torch.nn.init.xavier_uniform_(self.fully_connected.weight)
        self.fully_connected2 = nn.Linear(64, 1)
        # torch.nn.init.xavier_uniform_(self.fully_connected2.weight)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = x.view(-1, 16*NUM_NODES)
        x = self.fully_connected(x)
        x= F.relu(x)
        x = self.fully_connected2(x)

        return F.sigmoid(x)
    

print("CUDA!!!!!!!!!!!!!!", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# dataset = torch_geometric.datasets.NeuroGraphDataset(root='pyG', name='HCPGender')
# dataset = pytorch_dataset.AutismDataset(['NYU', 'USM', 'UM_1'], node_features='correlation', connectome_type='correlation')
dataset = pytorch_dataset.AutismDataset(['NYU'], node_features='correlation', connectome_type='correlation')
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_dataloader = DataLoader(train_dataset, 1, True)
test_dataloader = DataLoader(test_dataset, 1, True)
model = GCNModel(dataset.num_node_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)


model.train()
for epoch in tqdm(range(1, 51)):
    for data in train_dataloader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        out = out.squeeze(0)
        #print(out, data.y.float())
        loss = F.binary_cross_entropy(out, data.y.float())
        loss.backward()
        optimizer.step()

correct = 0
model.eval()
for data in test_dataloader:
    data = data.to(device)
    out = model(data)
    pred = out.round()
    correct += (pred == data.y.float())
    print(out.squeeze(0), data.y.float())

acc = int(correct) / len(test_dataloader.dataset)
print(f'Accuracy: {acc:.4f}')
