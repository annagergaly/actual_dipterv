import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric
import torch_geometric.nn
from tqdm import tqdm
import pytorch_dataset
from torch_geometric.loader import DataLoader
import wandb

NUM_NODES = 111
torch.random.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
wandb.init(project="dipterv", entity="anna-gergaly", name="hcp-gender", config = {
    "learning_rate": 0.0001,
    "test_partition": "separate",
})
all_sites = ["PITT", "OLIN", "OHSU", "SDSU", "TRINITY", "UM_2", "YALE", "CMU", "LEUVEN_1", "LEUVEN_2", "KKI", "STANFORD", "UCLA_1", "UCLA_2", "MAX_MUN", "CALTECH", "SBL", "NYU", "USM", "UM_1"]

sweep_config = {
    'name': 'leave_one_out_test',
    'method': 'grid',
    'metric': {
        'goal': 'maximize',
        'name': 'accuracy'
    },
    'parameters': {
        'epochs': {
            'values': [50]
        },
        'top_edges': {
            'values': [5, 10]
        },
        'leave_out': {
            'values': ["PITT", "YALE", "USM"]
        },
    }   
}
sweep_id = wandb.sweep(sweep_config, project="dipterv")


class GCNModel(nn.Module):
    def __init__(self, num_node_features):
        super(GCNModel, self).__init__()
        self.conv1 = torch_geometric.nn.GCNConv(num_node_features, 64)
        torch.nn.init.kaiming_uniform_(self.conv1.lin.weight)
        self.conv2 = torch_geometric.nn.GCNConv(64, 32)
        torch.nn.init.kaiming_uniform_(self.conv2.lin.weight)
        self.conv3 = torch_geometric.nn.GCNConv(32, 16)
        torch.nn.init.kaiming_uniform_(self.conv3.lin.weight)
        self.fully_connected = nn.Linear((NUM_NODES + 64 + 32 + 16), 64)
        torch.nn.init.kaiming_uniform_(self.fully_connected.weight)
        self.fully_connected2 = nn.Linear(64, 1)
        torch.nn.init.kaiming_uniform_(self.fully_connected2.weight)
        self.multi_aggr = torch_geometric.nn.aggr.MeanAggregation()
        self.batch_norm = torch_geometric.nn.BatchNorm((NUM_NODES + 64 + 32 + 16))

    def forward(self, data):
        x, edge_index, batches = data.x, data.edge_index, data.batch
        xs = [x]
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        xs.append(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        xs.append(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        xs.append(x)
        x = x.view(-1, 16*NUM_NODES)

        aggregated = []
        for xx in xs:
            aggregated.append(self.multi_aggr(xx, batches))

        #for a in aggregated:
        #    print(a.shape)

        aggregated = torch.cat(aggregated, dim=1)

        # print(aggregated.shape)
        if x.shape[0] == 1:
            x = aggregated
        else:
            x = self.batch_norm(aggregated)
        x = self.fully_connected(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fully_connected2(x)

        return F.sigmoid(x)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# dataset = torch_geometric.datasets.NeuroGraphDataset(root='pyG', name='HCPGender')
# dataset = pytorch_dataset.AutismDataset(['NYU', 'USM', 'UM_1'], node_features='correlation', connectome_type='correlation')
# dataset = pytorch_dataset.AutismDataset(wandb.config.dataset, node_features='correlation', connectome_type='correlation', top_edges=wandb.config.top_edges)

def main():
    run = wandb.init()
    # dataset = torch_geometric.datasets.NeuroGraphDataset(root='pyG', name='HCPGender')
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_sites = all_sites.copy()
    train_sites.remove(wandb.config.leave_out)
    dataset = pytorch_dataset.AutismDataset(train_sites, node_features='correlation', connectome_type='correlation', top_edges=wandb.config.top_edges)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    test_dataset = pytorch_dataset.AutismDataset([wandb.config.leave_out], node_features='correlation', connectome_type='correlation', top_edges=wandb.config.top_edges)
    train_dataloader = DataLoader(train_dataset, 1, True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, 1, True, drop_last=True)
    model = GCNModel(dataset.num_node_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

    model.train()
    for epoch in tqdm(range(1, wandb.config.epochs + 1)):
        avg_loss = 0
        for data in train_dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            out = model(data)
            out = out.squeeze()
            #print(out, data.y.float())
            loss = F.binary_cross_entropy(out, data.y.float().squeeze())
            avg_loss += loss.sum().item()
            wandb.log({"loss": loss})
            loss.backward()
            optimizer.step()
        avg_loss /= len(train_dataloader.dataset)
        avg_val_loss = 0
        for data in val_dataset:
            data = data.to(device)
            out = model(data)
            out = out.squeeze()
            loss = F.binary_cross_entropy(out, data.y.float().squeeze())
            avg_val_loss += loss.sum().item()
        avg_val_loss /= len(val_dataset)
        wandb.log({"avg_val_loss": avg_val_loss})
        wandb.log({"avg_loss": avg_loss})

    correct = 0
    model.eval()
    for data in test_dataloader:
        data = data.to(device)
        out = model(data)
        pred = out.round().squeeze()
        correct += (pred == data.y.float()).sum()
        # print(out.squeeze(0), data.y.float())

    acc = int(correct) / len(test_dataloader.dataset)
    print(f'Accuracy: {acc:.4f}')
    wandb.log({"accuracy": acc})



wandb.agent(sweep_id, function=main, count=120)
