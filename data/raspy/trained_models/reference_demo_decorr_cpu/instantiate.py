import yaml
import torch
import importlib.util
import numpy as np
from pathlib import Path
import os
import csv

def instantiate_model(config_file, model_architecture, model_file):
    # Load the configurations from the config file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Read in the model architecture
    spec = importlib.util.spec_from_file_location('Model', model_architecture)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Model = getattr(module, module.name)

    # Instantiate the model with params in config
    config_model = config['model']
    n_electrodes = 66 - len(config['data_preprocessor']['ch_to_drop'])
    config_dsop = config['dataset_generator']['dataset_operation']
    output_dim = len(config_dsop['selected_labels']) if not config_dsop['relabel'] else len(config_dsop['mapped_labels'])
    model = Model(config_model, output_dim, n_electrodes)

    # Load the model's state from the best trained fold
    device_ids = [0]
    if len(device_ids) > 1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    elif len(device_ids) == 1:
        device = torch.device(device_ids[0])
    else:
        device = torch.device('cpu')
    model.load_state_dict(torch.load(model_file, map_location=device))

    return model

def instantiate(params):

    # ==========  Parameters to extract  ==========
    path = Path(params['path'])
    path = path.parent / path.stem
    fold = params.get('fold', None)
    results_path = os.path.join(path, 'results.csv')

    #Get highest performing fold if not explicitly called
    if fold is None:
        with open(results_path) as results:
            results_dict = csv.DictReader(results)
            highest_val_acc = 0
            best_fold = None
            for row in results_dict:
                if float(row['Validation Acc']) > highest_val_acc:
                    best_fold = row['Validation Fold']
                    highest_val_acc = float(row['Validation Acc'])
            fold = int(best_fold)
            print("Fold Not specified: chose fold {fold}")

    config_file = path / 'config.yaml'
    model_architecture = path / 'EEGNet.py'
    model_file = path / str(fold)
    # ============================================

    model = instantiate_model(config_file, model_architecture, model_file)
    model.eval()

    return model



if __name__ == "__main__":
    # ==========  Parameters to Change  ==========
    path = './' #'/data/raspy/trained_models/wombats_dermatology_EEGNet_2023-07-22_S1_OL_1_RL/'
    config_file = path + 'config.yaml'
    model_architecture = path + 'EEGNet.py'
    model_file = path + '0'
    # ============================================

    model = instantiate_model(config_file, model_architecture, model_file)
    model.eval()
    
    x = torch.ones((1, 64, 100, 1)) #x = torch.ones((100, 64))
    y = model(x); print(y)
    y = model(x); print(y)
    y = model(x); print(y)
    y = model(x*1); print(y)
    y = model(x*3); print(y)
