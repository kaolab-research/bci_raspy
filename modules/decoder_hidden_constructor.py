'''derived from new_decoder module'''

from pathlib import Path
from importlib.machinery import SourceFileLoader
from modules.submodules.shared_utils.preprocessor import DataPreprocessor
from modules.submodules.shared_utils.utils import read_config
from modules.SJutil.DataStructure import deepDictUpdate
from scipy.signal import resample
import numpy as np
import torch
from torch import nn
import zipfile
############################################################
# create decoder from instantiate
############################################################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
decoder_path = Path(params['path'])
decoder_path = decoder_path.parent / decoder_path.stem

# check if model path exists & unzip
if decoder_path.exists(): pass
elif decoder_path.with_suffix('.zip').exists(): 
    with zipfile.ZipFile(decoder_path.with_suffix('.zip'),"r") as zip_ref:
        zip_ref.extractall(decoder_path.parent)
else: raise NameError("No such model at", decoder_path)

# instantiate model
instantiate = SourceFileLoader("instantiate",str(decoder_path / "instantiate.py")).load_module()
model = instantiate.instantiate(params)
model = model.to(device)

############################################################
# create DataPreprocessor from shared_utils
############################################################

config = read_config(str(Path(decoder_path / "config.yaml")))
dataPreprocessor = DataPreprocessor(config=deepDictUpdate(config["data_preprocessor"], params["data_preprocessor"]))
input_length = config['model']['window_length']
downsampled_length = int(config['model']['sampling_frequency'] * config['model']['window_length'] / 1000)

# decoder_output will always be 5 length
decoder_output[:] = np.zeros(decoder_output.shape)
