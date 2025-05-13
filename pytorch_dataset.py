import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch.nn.functional import normalize

PREPROCESS_TYPE = f'Outputs/cpac/filt_noglobal/rois_ho'
ROOT_PATH = f'C:/Users/Anna/Documents/actual_dipterv/data'

def load_site(site, connectome_type='correlation'):
    
    autistic = []
    control = []
    for roi_file in os.listdir(f'{ROOT_PATH}/{site}/{PREPROCESS_TYPE}'):
        data = np.loadtxt(f'{ROOT_PATH}/{site}/{PREPROCESS_TYPE}/{roi_file}')
        data = data - data.mean(axis=0)
        data = data / (data.std(axis=0) + 1e-8)  # Avoid division by zero
        data = torch.tensor(data)
        autistic.append(data.T)

    for roi_file in os.listdir(f'{ROOT_PATH}/{site}_control/{PREPROCESS_TYPE}'):
        data = np.loadtxt(f'{ROOT_PATH}/{site}_control/{PREPROCESS_TYPE}/{roi_file}')
        data = data - data.mean(axis=0)
        data = data / (data.std(axis=0) + 1e-8)  # Avoid division by zero
        data = torch.tensor(data)
        control.append(data.T)

    #TODO padding or trimming
    # autistic = np.stack(autistic)
    # control = np.stack(control)

    autistic_connectomes = np.load(f'{ROOT_PATH}/{site}/connectomes_{connectome_type}.npy', allow_pickle=True)
    control_connectomes = np.load(f'{ROOT_PATH}/{site}_control/connectomes_{connectome_type}.npy', allow_pickle=True)
    autistic_connectomes = torch.tensor(autistic_connectomes, dtype=torch.float)
    control_connectomes = torch.tensor(control_connectomes, dtype=torch.float)
    return (autistic, control, autistic_connectomes, control_connectomes)

class AutismDataset(Dataset):
    def __init__(self, sites, connectome_type='correlation', node_features='correlation', top_edges=5):
        self.top_edges = top_edges
        self.roi_timeseries = []
        self.node_features = []
        self.connectomes = []
        self.labels = []
        for site in sites:
            (autistic, control, autistic_connectomes, control_connectomes) = load_site(site, connectome_type)
            self.roi_timeseries = self.roi_timeseries + autistic
            self.roi_timeseries = self.roi_timeseries + control
            self.labels = self.labels + [1 for _ in autistic]
            self.labels = self.labels + [0 for _ in control]
            autistic_connectomes = np.vsplit(autistic_connectomes, len(autistic))
            autistic_connectomes = [sample.squeeze() for sample in autistic_connectomes]
            control_connectomes = np.vsplit(control_connectomes, len(control))
            control_connectomes = [sample.squeeze() for sample in control_connectomes]
            self.connectomes = self.connectomes + autistic_connectomes
            self.connectomes = self.connectomes + control_connectomes
        self.length = len(self.labels)
        print(self.roi_timeseries[0].shape)
        print(self.connectomes[0].shape)
        match node_features:
            case 'roi_timeseries':
                self.node_features = self.roi_timeseries
            case 'average':
                self.node_features = [torch.diag_embed(torch.mean(sample, dim=1, dtype=torch.float)) for sample in self.roi_timeseries]
            case 'max':
                self.node_features = [torch.diag_embed(torch.max(sample, dim=1, dtype=torch.float)) for sample in self.roi_timeseries]
            case 'min':
                self.node_features = [torch.diag_embed(torch.min(sample, dim=1, dtype=torch.float)) for sample in self.roi_timeseries]
            case 'identity':
                self.node_features = [torch.eye(sample.shape[0], dtype=torch.float) for sample in self.roi_timeseries]
            case 'correlation':
                self.node_features = self.connectomes
        self.num_node_features = self.roi_timeseries[0].shape[0]
        self.num_classes = 2
        self.num_nodes = self.connectomes[0].shape[0]



    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        threshold = torch.quantile(self.connectomes[idx], (100-self.top_edges)/100)
        adj = self.connectomes[idx].clamp(0, 1)
        adj[adj < threshold] = 0
        edge_index, edge_weight = dense_to_sparse(adj)
        x = self.node_features[idx]
        y = torch.tensor(self.labels[idx])

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=y,
            num_nodes=adj.shape[0]
        )
    
    
    #(self.roi_timeseries[idx], self.connectomes[idx], self.labels[idx])