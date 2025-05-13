import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric
import torch_geometric.nn
from tqdm import tqdm

class GCNModel(nn.Module):
    def __init__(self, num_node_features):
        super(GCNModel, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(num_node_features, 32)
        # torch.nn.init.xavier_uniform_(self.conv1.lin.weight)
        self.conv2 = torch_geometric.nn.GCNConv(32, 16)
        # torch.nn.init.xavier_uniform_(self.conv2.lin.weight)
        self.fully_connected = nn.Linear(16*1000, 64)
        # torch.nn.init.xavier_uniform_(self.fully_connected.weight)
        self.fully_connected2 = nn.Linear(64, 1)
        # torch.nn.init.xavier_uniform_(self.fully_connected2.weight)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = x.view(-1, 16*1000)
        x = self.fully_connected(x)
        x= F.relu(x)
        x = self.fully_connected2(x)

        return F.sigmoid(x)
    

print("CUDA!!!!!!!!!!!!!!", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
dataset = torch_geometric.datasets.NeuroGraphDataset(root='pyG', name='HCPGender').to(device).shuffle()
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
model = GCNModel(dataset.num_node_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)


model.train()
for epoch in tqdm(range(1, 21)):
    for data in train_dataset:
        optimizer.zero_grad()
        out = model(data)
        out = out.squeeze(0)
        #print(out, data.y.float())
        loss = F.binary_cross_entropy(out, data.y.float())
        loss.backward()
        optimizer.step()

correct = 0
model.eval()
for data in test_dataset:
    out = model(data)
    pred = out.round()
    correct += (pred == data.y.float())
    print(out.squeeze(0), data.y.float())

acc = int(correct) / len(test_dataset)
print(f'Accuracy: {acc:.4f}')
