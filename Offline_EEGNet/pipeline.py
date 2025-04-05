# # TODO list -----------------------------:
# interact with the sql table
# version 3

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from shared_utils import utils
from shared_utils.dataset import create_dataset
from EEGData import EEGData
from torch.utils.data import DataLoader
from EEGNet import EEGNet
from train import train, test
import csv
import git
import sqlite3
import pandas as pd
import random
import shutil
from utils import create_confusion_matrix 
from flatten_dict import flatten
import pickle
from torch.utils.tensorboard import SummaryWriter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
writer = SummaryWriter(log_dir='runs/test-eegnet')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# ==================== Preparation ====================
# Read in config file
yaml_file = sys.argv[1]
config = utils.read_config(yaml_file)

# Write into sql dataset
to_sql = flatten(config)
repo = git.Repo(search_parent_directories=True)
to_sql['git_hash'] = repo.head.object.hexsha
conn = sqlite3.connect('/data/raspy/sql/sql_eeg.db') if config['train_on_server'] else None
# TODO: lack of adding this record into sql table

# Generate model name
model_name = utils.model_namer(conn, config['train_on_server'], config['model_arch_dir'].split('/')[-1][:-3])

model_dir = config['model_dir'] + model_name + '_' + '_'.join(config['data_names']) + '/'
print('============================================================')
print(f'The name of this model is: {model_name}, from config {yaml_file}')
print(f'at model_dir {model_dir}')
print('============================================================')

# Set random seed for reproducibility
if config['random_seed']:
    seed = config['random_seed']
else:
    random.seed(str(config['data_names']))          # use data names as seed for random module
    seed = random.randint(0, 2**32-1)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'



# ============== h5 Dataset Creation ===============
h5_path = config['h5_dir'] + model_name + '.h5'
create_dataset(config, h5_path=h5_path)


# ==================== Training ====================
config_train = config['training']
config_model = config['model']


os.makedirs(model_dir)

# Save files
shutil.copy(config['model_arch_dir'], model_dir)
#shutil.copy(config['config_dir'], model_dir)
shutil.copy(yaml_file, model_dir + 'config.yaml')
shutil.copy('./instantiate.py', model_dir)

# Train and validate the model of each fold
losses = {}
best_fold = ''

with open(model_dir + "/results.csv", mode='w') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Validation Fold", "Training Acc", "Validation Acc"])

all_pred_labels, all_true_labels = [], []
for validation_fold in range(config_train['num_folds']):
    best_acc = 0

    train_folds = list(range(config_train['num_folds']))
    train_folds.pop(validation_fold)

    train_dataset = EEGData(h5_path, train_folds)
    train_dataset = DataLoader(train_dataset, 
                               batch_size  = config_train['train_batch_size'],
                               shuffle = config_train['train_shuffle'],
                               drop_last = config_train['train_drop_last'],
                               num_workers = config_train['train_num_workers'],
                               prefetch_factor = config_train['train_prefetch_factor'])

    validation_dataset = EEGData(h5_path, [validation_fold], train=False)
    validation_dataset = DataLoader(validation_dataset, 
                                    batch_size  = config_train['val_batch_size'],
                                    shuffle = config_train['val_shuffle'],
                                    drop_last = config_train['val_drop_last'],
                                    num_workers = config_train['val_num_workers'],
                                    prefetch_factor = config_train['val_prefetch_factor'])

    n_electrodes = 66 - len(config['data_preprocessor']['ch_to_drop'])
    config_dsop = config['dataset_generator']['dataset_operation']
    output_dim = len(config_dsop['selected_labels']) if not config_dsop['relabel'] else len(config_dsop['mapped_labels'])
    model = EEGNet(config_model, output_dim, n_electrodes)

    device_ids = config['device_id'] if config['train_on_server'] else [0]
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    trained_EEGNet, train_stats, best_model_index, train_losses, val_losses, pred_labels, true_labels = train(model, train_dataset, validation_dataset, config_train, writer)

    torch.save(trained_EEGNet.state_dict(), model_dir + str(validation_fold))  # saves the best model

    # write data to CSV file (in append mode)
    with open(model_dir + '/results.csv', mode='a') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([validation_fold,
                            train_stats['acc'][best_model_index], 
                            train_stats['val_acc'][best_model_index]])

    if train_stats['val_acc'][best_model_index] > best_acc:
        best_acc = train_stats['val_acc'][best_model_index]
        best_model = trained_EEGNet
        best_fold = validation_fold
        
    losses[validation_fold] = {}
    losses[validation_fold]['training'] = train_losses
    losses[validation_fold]['val'] = val_losses


    # create confusion matrix for each
    os.makedirs(model_dir+"/confusion_matrix/",exist_ok=True)
    confusion_matrix_file_path = model_dir+f"/confusion_matrix/cm{validation_fold}.jpg"
    create_confusion_matrix(pred_labels[best_model_index],true_labels[best_model_index],confusion_matrix_file_path)

    all_pred_labels.extend(pred_labels[best_model_index])
    all_true_labels.extend(true_labels[best_model_index])


# Save results
if not os.path.exists('results'):
    os.makedirs('results/')
file_path = './results/losses.pickle'
with open(file_path, 'wb') as file:
    pickle.dump(losses, file)

with open('./results/labels.txt', 'w') as out:
    for i, j in zip(all_pred_labels, all_true_labels):
        print(i, j, file=out)

writer.close()
os.remove(h5_path)
shutil.copy('./results/losses.pickle', model_dir + 'losses.pickle')
shutil.copy('./results/labels.txt', model_dir + 'labels.txt')


results = pd.read_csv(model_dir + "/results.csv")
# to_sql['name'] = model_name
# to_sql['avg_val_acc'] = results['Validation Acc'].mean()
# to_sql['model_path'] = model_dir
# to_sql = pd.DataFrame(to_sql, index=[0])

print('============================================================')
print(f'The name of this model is: {model_name}, from config {yaml_file}')
print(f'at model_dir {model_dir}')
print('============================================================')
