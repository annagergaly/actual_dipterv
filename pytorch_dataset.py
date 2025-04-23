import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

PREPROCESS_TYPE = f'Outputs/cpac/filt_noglobal/rois_ho'
ROOT_PATH = f'C:/Users/Anna/Documents/actual_dipterv/data'

def load_site(site):
    
    autistic = []
    control = []
    for roi_file in os.listdir(f'{ROOT_PATH}/{site}/{PREPROCESS_TYPE}'):
        data = np.loadtxt(f'{ROOT_PATH}/{site}/{PREPROCESS_TYPE}/{roi_file}')
        autistic.append(data.T)

    for roi_file in os.listdir(f'{ROOT_PATH}/{site}_control/{PREPROCESS_TYPE}'):
        data = np.loadtxt(f'{ROOT_PATH}/{site}_control/{PREPROCESS_TYPE}/{roi_file}')
        control.append(data.T)

    #TODO padding or trimming
    # autistic = np.stack(autistic)
    # control = np.stack(control)

    autistic_connectomes = np.load(f'{ROOT_PATH}/{site}/connectomes.npy', allow_pickle=True)
    control_connectomes = np.load(f'{ROOT_PATH}/{site}_control/connectomes.npy', allow_pickle=True)
    return (autistic, control, autistic_connectomes, control_connectomes)

class AutismDataset(Dataset):
    def __init__(self, sites):
        self.roi_timeseries = []
        self.connectomes = []
        self.labels = []
        for site in sites:
            (autistic, control, autistic_connectomes, control_connectomes) = load_site(site)
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
        self.num_node_features = self.roi_timeseries[0].shape[1]
        self.num_classes = 2
        self.num_nodes = self.connectomes[0].shape[0]



    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        adj = torch.tensor(self.connectomes[idx])
        edge_index, edge_weight = dense_to_sparse(adj)
        x = torch.tensor(self.roi_timeseries[idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx])

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=y,
            num_nodes=adj.shape[0]
        )
    
    
    #(self.roi_timeseries[idx], self.connectomes[idx], self.labels[idx])