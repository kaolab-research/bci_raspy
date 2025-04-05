import numpy as np
import scipy.io as sio
import os
import scipy
import scipy.signal
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

import argparse
import ast
import multiprocessing as mp
import itertools
from copy import deepcopy

import pipeline_kf_func
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

DEFAULT_metaconfig = {'config': {}, 'model_name': None, 'data_name': None, 'verbosity': 1} # for reference. This is not valid because config is empty

def train_from_metaconfigs(metaconfigs):
    '''
    metaconfigs should be an iterable of metaconfigs
    A metaconfig would be, by default, {'config': {}, 'model_name': None, 'data_name': None, 'verbosity': 1}
    '''
    for metaconfig in metaconfigs:
        pipeline_kf_func.pipeline(metaconfig['config'], model_name=metaconfig['model_name'], data_name=metaconfig['data_name'], verbosity=metaconfig['verbosity'],
                                 train_kf=True)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', default='(2,)', help='int or tuple of devices to use')
    parser.add_argument('--workers', default='1', help='number of workers per device. Either int or tuple')
    args = parser.parse_args()

    if isinstance(args.devices, str):
        DEVICES = ast.literal_eval(args.devices)
    else:
        DEVICES = args.devices
    if isinstance(DEVICES, int):
        DEVICES = (DEVICES,)
    if isinstance(args.workers, str):
        WORKERS = ast.literal_eval(args.workers)
    else:
        WORKERS = args.workers
    if isinstance(WORKERS, int):
        WORKERS = (WORKERS,)*len(DEVICES)
    
    
    t00 = time.time()
    procs = {}
    total_workers = np.sum(WORKERS)
    workers_tuples = []
    for device_no, (device_id, n_workers) in enumerate(zip(DEVICES, WORKERS)):
        # device_no is the index within the DEVICES tuple
        for worker_no in range(n_workers):
            workers_tuples.append((device_id, worker_no))
        pass
    print('workers_tuples', workers_tuples, total_workers)
    subjects_dict = {worker_tuple: [] for worker_tuple in workers_tuples}
    
    #class_ids = [-127, -126, 126, 127, 125, 5]
    #class_ids = [-127, -126, 120, 125, 5]
    #class_ids = [5, 50, 51]
    class_ids = [4, 5, 123, 125, 126, 127] # for '2023-09-02_S1_OL_1'
    n_classes = 4
    class_id_combinations = list(itertools.combinations(class_ids, n_classes))
    
    config_template_fname = 'config-template.yaml'
    with open(config_template_fname, 'r') as config_file:
        config_template = yaml.load(config_file, Loader=Loader)
    #config_template['data_names'] = ['2023-10-16_A2_OL_1']
    #config_template['data_names'] = ['2023-08-15_A2_OL_1']
    #config_template['data_names'] = ['2023-08-15_A2_OL_2']
    config_template['data_names'] = ['2023-09-02_S1_OL_1']
    config_template['dataset_generator']['dataset_operation']['relabel'] = True
    config_template['model']['num_temporal_filters']
    config_template['model']['num_spatial_filters']
    
    #data_name = '2023-10-16_A2_OL_1_L1000'
    #h5_dir = '/data/raspy/preprocessed_data/'
    #h5_path = h5_dir + data_name + '.h5'
    #pipeline_kf_func.create_dataset(config_template, h5_path=h5_path)
    
    #worker_idx = 0
    metaconfigs = []
    for combo_no, combination in enumerate(class_id_combinations):
        config = deepcopy(config_template)
        item = {f'class{class_no}': [combination[class_no]] for class_no in range(n_classes)}
        config['dataset_generator']['dataset_operation']['mapped_labels'] = item
        print(config['dataset_generator']['dataset_operation']['mapped_labels'])
        model_name = f'combos-{combo_no}_vanilla_L1000'
        data_name = None
        metaconfig = {
            'config': config,
            'model_name': model_name,
            'data_name': data_name,
            'verbosity': 0,
        }
        metaconfigs.append(metaconfig)
    #for subject in range(1, n_subjects+1):
    #    subjects_dict[workers_tuples[worker_idx]].append(subject)
    #    worker_idx = (worker_idx + 1) % total_workers
    #print('subjects_dict', subjects_dict)
    
    #for worker_tuple in workers_tuples:
    #    subjects = subjects_dict[worker_tuple]
    #    device_id = worker_tuple[0]
    #    procs[worker_tuple] = mp.Process(target=train_from_metaconfigs, args=(subjects, device_id), name=str(worker_tuple))
    
    
    for worker_idx, worker_tuple in enumerate(workers_tuples):
        worker_metaconfigs = metaconfigs[worker_idx::len(workers_tuples)]
        for metaconfig in worker_metaconfigs:
            metaconfig['config']['device_id'] = [worker_tuple[0]] # device_id. should be integer
        procs[worker_tuple] = mp.Process(target=train_from_metaconfigs, args=(worker_metaconfigs,), name=str(worker_tuple))
        pass
    
    for _, proc in procs.items():
        proc.start()
    for _, proc in procs.items():
        proc.join()
    print(f'done in {time.time()-t00} seconds')
    
    