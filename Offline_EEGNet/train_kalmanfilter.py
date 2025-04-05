
import numpy as np
from kalman_filter import KalmanFilter
import train

def gen_one_hot(y, dtype='float', max_val=4):
    out = np.zeros((len(y), max_val+1), dtype=dtype)
    out[np.arange(len(y)), y] = 1
    return out

def train_kf_from_ol(model, dsets, RLUD_idx):
    criterion = train.OneHotMSE()
    # each item of outs has [logits_collect, hidden_states_collect, labels_collect, loss_collect], in order
    model_outputs = [train.collect_outputs(model, dset, criterion, return_numpy=True) for dset in dsets]
    hidden_states = np.concatenate([model_output[1] for model_output in model_outputs], axis=0)
    labels = np.concatenate([model_output[2] for model_output in model_outputs], axis=0)
    print('kf labels', len(labels))
    avg_mse = (1/criterion.gamma_sq)*np.mean(np.concatenate([model_output[3] for model_output in model_outputs], axis=0))
    
    obs_dim = hidden_states.shape[-1]
    print('hdim:', hidden_states.shape[-1])
    rect_vels = gen_one_hot(labels)
    kf_states = np.zeros((rect_vels.shape[0], 7))
    kf_states[:, -1] = 1.0
    kf_states[:, 2:6] = rect_vels[:, RLUD_idx] # RLUD_idx corresponds to right, left, up, down of labels.
    
    kf_model = KalmanFilter(state_dim=7, obs_dim=obs_dim)
    dt_s = 0.050 # in seconds. To-do: read from config?
    A_gain = 1.0
    kf_model.A = kf_model.get_A(dt_s, gain=A_gain)
    kf_model.set_W_diag(avg_mse) # set velocity diagonals of W to the average MSE
    kf_model.fit(kf_states, hidden_states, max_iter=100) # This does not fit A or W, since prev_states is not given
    return kf_model

