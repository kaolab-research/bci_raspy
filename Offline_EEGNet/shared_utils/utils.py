import sys
import numpy as np
from scipy import signal
import ast
import random
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import pandas as pd
import sqlite3
import pathlib

class DumperWithIndent(yaml.Dumper):
    '''https://stackoverflow.com/questions/25108581/python-yaml-dump-bad-indentation
    Indents list item lines
    '''
    def increase_indent(self, flow=False, indentless=False):
        return super(DumperWithIndent, self).increase_indent(flow, False)
def dump_to_yaml(yaml_data, fname):
    with open(fname, 'w') as f:
        yaml.dump(yaml_data, f, Dumper=DumperWithIndent, default_flow_style=False, sort_keys=False)
    return

def model_namer(sql_conn, train_on_server, model_arch_name):
    '''Generates a unique two word name based on the inbuilt unix dictionary. 
    
    Parameters
    ----------
    sql_conn: sqlite3.Connection
    train_on_server: bool
    model_arch_name: string
        Name of the model architecture.

    Returns
    -------
    new_name: string
        Generated two-word model name with model architecture's name, e.g. wooden_jazz_EEGNet.
    '''

    table = 'EEGNet'
    # get previously used names
    if train_on_server == True:
        used_names = pd.read_sql(f'SELECT * FROM {table}', sql_conn)
        used_names = list(used_names.name)
    
    # import word list from unix install
    with open(pathlib.Path(__file__) / '..' / 'words') as f:
        words = f.read().splitlines()
    words = [word.lower() for word in words if "'" not in word]
    
    if train_on_server == True:
        unique_name = False
        while not unique_name:
            new_name = random.choice(words) + '_' + random.choice(words)
            new_name = new_name + '_' + model_arch_name
            if new_name not in used_names:
                unique_name = True
    else:                                                                   # cannot check duplicate names for local training
        new_name = random.choice(words) + '_' + random.choice(words)

    return new_name



def decide_kind(data_name):
    '''Decide which kind of dataset it is based on the information in the data_name.

    Parameters
    ----------
    data_name: string
        The name of the data. Should be in format: YYYY-MM-DD_SUBJECT_KIND_SESSIONID_NOTES.

    Returns
    -------
    kind: string
        Either 'OL' or 'CL'.
    '''
    
    if 'OL' in data_name:
        kind = 'OL'
    elif 'CL' in data_name:
        kind = 'CL'
    else:
        raise ValueError(f'The dataset {data_name} doesn\'t indicate whether it is OL or CL. Please check it.')
    
    return kind



def read_config(yaml_name):
    '''Read in the information from the yaml setting file where stores which data to use.

    Parameters
    ----------
    yaml_name: str
        The yaml file name. Example: settings.yaml

    Returns
    -------
    config: dict
        A dict of information in the assigned yaml file.
    '''

    yaml_file = open(yaml_name, 'r')
    config = yaml.safe_load(yaml_file)
    yaml_file.close()
    return config



def read_data_file_to_dict(filename, return_dict=True):
    '''Read in the information in .bin file into a dict.

    Parameters
    ----------
    filename: str
        Path to .bin file.
    return_dict: bool
        Whether return a dictionary format or an array format.

    Returns
    -------
    If return dict and it's eeg data:
        eeg_data: dict
            'eegbuffersignal': 2-d array with shape (n_samples, n_electrodes)
                Raw data collected with sampling rate as 1000 Hz. Already applied bandpass filter: 4 - 90 Hz.
            'databuffer': 2-d array with shape (n_samples, n_electrodes)
                Filtered raw data with sampling rate as 1000 Hz. Already applied bandpass filter: 4 - 40 Hz.
            'task_step': 1-d array with shape (n_samples,)
                Record the sample indices in the task data that each eeg data corresponds to. Each element is a number in [0, n_task_samples].
                E.g.: (array([  252,   253,   255, ..., 75038, 75039, 75040], dtype=int32), array([ 2, 15, 43, ..., 20, 21, 20]))
            'time_ns': 1-d array with shape (n_samples,)
                The absolute time in nanoseconds.
            'name': str, 'eeg'
            'dtypes': list, ['66<f4', '66<f4', '<i4', '<i8']

    If return dict and it's task data:
        task_data: dict
            'state_task': 1-d array with shape (n_task_samples,)
                State we set, like [-1,  0,  1,  2,  3,  4]. Sampling rate is 50 Hz.
            'decoder_output': 2-d array with shape (n_task_samples, n_class)
                For OL, it's all zero.
            'decoded_pos': 2-d array with shape (n_task_samples, 2)
                For OL, it's all zero.
            'target_pos': 2-d array with shape (n_task_samples, 2)
                Record the target position.
                    [[-0.85  0.  ]
                     [ 0.   -0.85]
                     [ 0.    0.85]
                     [ 0.85  0.  ]]
            'eeg_step': 1-d array with shape (n_task_samples,)
                Record the sample indices in the eeg data that each task data corresponds to. Each element is a number in [0, n_samples].
            'time_ns': 1-d array with shape (n_task_samples,)
                The absolute time in nanoseconds.
            'name': str, 'task'
            'dtypes': list, ['|i1', '5<f4', '2<f4', '2<f4', '<i4', '<i8']

    If not return dict and it's eeg data:
        data: 1-d array with shape (n_samples,)
            Each row is a numpy.void with length 4, coresponding to 'eegbuffersignal' (in shape (n_electrodes,)), 'databuffer', 'task_step', 'time_ns'.

    If not return dict and it's task data:
        data: 1-d array with shape (n_task_samples,)
            Each row is a numpy.void with length 6, coresponding to 'state_task', 'decoder_output', 'decoded_pos', 'target_pos', 'eeg_step', 'time_ns'.
    '''

    with open(filename, 'rb') as openfile:
        name = openfile.readline().decode('utf-8').strip()
        keys = openfile.readline().decode('utf-8').strip()
        dtypes = openfile.readline().decode('utf-8').strip()
        shapes = None

        if len(dtypes.split('$')) == 2:             # shapes can be indicated with a $ to separate.
            dtypes, shapes = dtypes.split('$')
            dtypes = dtypes.strip()
            shapes = ast.literal_eval(shapes.strip())
        
        keys = keys.split(',')
        dtypes = dtypes.split(',')
        if shapes is None:
            data = np.fromfile(openfile, dtype=[item for item in zip(keys, dtypes)])
        else:
            data = np.fromfile(openfile, dtype=[item for item in zip(keys, dtypes, shapes)])
        if not return_dict:
            return data
        data_dict = {key: data[key] for key in keys}
        data_dict['name'] = name
        data_dict['dtypes'] = dtypes
    return data_dict


if __name__ == "__main__":

    # put in yaml file name as input (i.e config.yaml)
    yaml_file = sys.argv[1]
    config = read_config(yaml_file)

    # Generate model name
    conn = sqlite3.connect('/data/raspy/sql/sql_eeg.db')
    model_name = model_namer(conn, train_on_server=True, model_arch_name="EEGNet")

    # Preprocessing example
    dataName = '2023-07-22_S1_OL_1_RL'
    dataDir = '/data/raspy/'
    eeg_data = read_data_file_to_dict(dataDir + dataName + "/eeg.bin")
    task_data = read_data_file_to_dict(dataDir + dataName + "/task.bin")
    kind = decide_kind(dataName)

    print("kind", kind)
    print("EEG_Data",eeg_data["databuffer"].shape)
    print("random model name", model_name)