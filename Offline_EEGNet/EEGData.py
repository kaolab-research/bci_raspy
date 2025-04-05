import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
import h5py


class EEGData(Dataset):
    '''Stores the samples and their corresponding labels.

    Example
    -------
    train_dataset = EEGData(h5_file, train_folds)
    validation_dataset = EEGData(h5_file, [validation_fold], train=False)
    '''

    def __init__(self, h5_file, folds, train=True):
        '''Get datasets from the data file.

        Args
        ----
        h5_file: str
            The path to the .h5 data file.
        folds: list
            List of fold indices we use to generate this dataset.
        '''

        self.file = h5py.File(h5_file, 'r')
        self.folds = folds
        self.train = train

        suffix = ['_trials', '_labels']
        if not self.train:
            suffix = ['_val_trials', '_val_labels']
        self.trials_folds = [str(fold)+suffix[0] for fold in folds]
        self.labels_folds = [str(fold)+suffix[1] for fold in folds]

        self.lens = [self.file[i].shape[0] for i in self.labels_folds]
        self.cumusum = np.cumsum(self.lens)

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, idx):
        fold_id = np.searchsorted(self.cumusum, idx, side='right')
        trial_id = idx if fold_id == 0 else idx - self.cumusum[fold_id - 1]

        trial = self.file[self.trials_folds[fold_id]][trial_id]
        trial = torch.from_numpy(trial).type(torch.float)
        label = self.file[self.labels_folds[fold_id]][trial_id]
        label = torch.tensor(label).type(torch.long)
        return trial, label
    
class CachedEEGData(Dataset):
    '''Stores the samples and their corresponding labels.

    Example
    -------
    train_dataset = EEGData(h5_file, train_folds)
    validation_dataset = EEGData(h5_file, [validation_fold], train=False)
    '''

    def __init__(self, h5_file, folds, train=True):
        '''Get datasets from the data file.

        Args
        ----
        h5_file: str
            The path to the .h5 data file.
        folds: list
            List of fold indices we use to generate this dataset.
        '''
        print('Using CachedEEGData')

        self.file = h5py.File(h5_file, 'r')
        self.folds = folds
        self.train = train

        suffix = ['_trials', '_labels']
        if not self.train:
            suffix = ['_val_trials', '_val_labels']
        self.trials_folds = [str(fold)+suffix[0] for fold in folds]
        self.labels_folds = [str(fold)+suffix[1] for fold in folds]
        
        # Used to cache trials once they are loaded
        self.file_cache = {key: dict() for key in self.trials_folds + self.labels_folds}

        self.lens = [self.file[i].shape[0] for i in self.labels_folds]
        self.cumusum = np.cumsum(self.lens)

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, idx):
        fold_id = np.searchsorted(self.cumusum, idx, side='right')
        trial_id = idx if fold_id == 0 else idx - self.cumusum[fold_id - 1]
        
        if trial_id in self.file_cache[self.trials_folds[fold_id]]:
            trial = self.file_cache[self.trials_folds[fold_id]][trial_id]
            label = self.file_cache[self.labels_folds[fold_id]][trial_id]
            #print('from cache', trial.shape, label.shape)
        else:
            trial = self.file[self.trials_folds[fold_id]][trial_id]
            trial = torch.from_numpy(trial).type(torch.float)
            label = self.file[self.labels_folds[fold_id]][trial_id]
            label = torch.tensor(label).type(torch.long)
            
            self.file_cache[self.trials_folds[fold_id]][trial_id] = trial.clone()
            self.file_cache[self.labels_folds[fold_id]][trial_id] = label.clone()
            #print('from file', trial.shape, label.shape)
        return trial, label

#class EEGData(Dataset):
#    '''Stores the samples and their corresponding labels.
#
#    Example
#    -------
#    train_dataset = EEGData(h5_file, train_folds)
#    validation_dataset = EEGData(h5_file, [validation_fold], train=False)
#    '''
#
#    def __init__(self, h5_file, folds, train=True):
#        '''Get datasets from the data file.
#
#        Args
#        ----
#        h5_file: str
#            The path to the .h5 data file.
#        folds: list
#            List of fold indices we use to generate this dataset.
#        '''
#
#        self.file = h5py.File(h5_file, 'r')
#        self.folds = folds
#        self.train = train
#        
#
#        suffix = ['_trials', '_labels']
#        if not self.train:
#            suffix = ['_val_trials', '_val_labels']
#        self.trials_folds = [str(fold)+suffix[0] for fold in folds]
#        self.labels_folds = [str(fold)+suffix[1] for fold in folds]
#        
#        # Used to cache trials once they are loaded
#        self.file_cache = {key: dict() for key in self.trials_folds + self.labels_folds}
#
#        self.lens = [self.file[i].shape[0] for i in self.labels_folds]
#        self.cumusum = np.cumsum(self.lens)
#
#    def __len__(self):
#        return sum(self.lens)
#
#    def __getitem__(self, idx):
#        fold_id = np.searchsorted(self.cumusum, idx, side='right')
#        trial_id = idx if fold_id == 0 else idx - self.cumusum[fold_id - 1]
#        
#        print(idx)
#        try:
#            trial = self.file_cache[self.trials_folds[fold_id]][trial_id]
#            label = self.file_cache[self.labels_folds[fold_id]][trial_id]
#            print(trial.shape, label.shape)
#        except:
#            trial = self.file[self.trials_folds[fold_id]][trial_id]
#            trial = torch.from_numpy(trial).type(torch.float)
#            label = self.file[self.labels_folds[fold_id]][trial_id]
#            label = torch.LongTensor(label)
#            self.file_cache[self.trials_folds[fold_id]][trial_id] = trial
#            self.file_cache[self.labels_folds[fold_id]][trial_id] = label
#        return trial, label
