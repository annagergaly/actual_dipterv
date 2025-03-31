import os
import pandas as pd
import numpy as np
from torch import Dataset

PREPROCESS_TYPE = f'Outputs/cpac/filt_noglobal/rois_ho'
ROOT_PATH = f'C:/Egyetem/Msc/actual_dipterv/data'

def load_site(site):
    
    autistic = []
    control = []
    for roi_file in os.listdir(f'{ROOT_PATH}/{site}/{PREPROCESS_TYPE}'):
        data = np.loadtxt(roi_file)
        autistic.append(data)

    for roi_file in os.listdir(f'{ROOT_PATH}/{site}_control/{PREPROCESS_TYPE}'):
        data = np.loadtxt(roi_file)
        control.append(data)

    #TODO padding or trimming
    autistic = np.stack(autistic)
    control = np.stack(control)

    autistic_connectomes = np.load(f'{ROOT_PATH}/{site}/connectomes.npy')
    control_connectomes = np.load(f'{ROOT_PATH}/{site}_control/connectomes.npy')
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



    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.roi_timeseries[idx], self.connectomes[idx], self.labels[idx])