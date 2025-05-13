from nilearn import connectome
import os
import numpy as np
import pandas as pd

PREPROCESS_TYPE = f'Outputs/cpac/filt_noglobal/rois_ho'
ROOT_PATH = f'C:/Users/Anna/Documents/actual_dipterv/data'


def load_site(site):
    autistic = []
    autistic_file_names = []
    control = []
    control_file_names = []
    for roi_file in os.listdir(f'{ROOT_PATH}/{site}/{PREPROCESS_TYPE}'):
        data = np.loadtxt(f'{ROOT_PATH}/{site}/{PREPROCESS_TYPE}/{roi_file}')
        data = data - data.mean(axis=0)
        data = data / (data.std(axis=0) + 1e-8)  # Avoid division by zero
        autistic_file_names.append(roi_file[:-3])
        autistic.append(data)

    for roi_file in os.listdir(f'{ROOT_PATH}/{site}_control/{PREPROCESS_TYPE}'):
        data = np.loadtxt(f'{ROOT_PATH}/{site}_control/{PREPROCESS_TYPE}/{roi_file}')
        data = data - data.mean(axis=0)
        data = data / (data.std(axis=0) + 1e-8)  # Avoid division by zero
        control_file_names.append(roi_file[:-3])
        control.append(data)

    print(autistic[0].shape)
    autistic = np.stack(autistic)
    control = np.stack(control)
    return (autistic, autistic_file_names, control, control_file_names)

if __name__ == '__main__':
    site = 'UM_1'
    (autistic, a_files, control, c_files) = load_site(site)
    estimator = connectome.ConnectivityMeasure(kind='correlation')

    autistic_connectomes = estimator.fit_transform(autistic)
    control_connectomes = estimator.fit_transform(control)

    np.save(f'{ROOT_PATH}/{site}/connectomes_correlation', autistic_connectomes)
    np.save(f'{ROOT_PATH}/{site}_control/connectomes_correlation', control_connectomes)

