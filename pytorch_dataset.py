import os
import torch
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
    return (autistic, control)

class AutismDataset(Dataset):
    def __init__(self, sites):
        for site in sites:
            load_site(site)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):