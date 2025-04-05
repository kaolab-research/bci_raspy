
import numpy as np
from kalman_filter import KalmanFilter

def gen_default_state():
    out = np.zeros(7)
    out[-1] = 1.0
    return out
def relu(x):
    return x*(x > 0)
def rectify_vels(vels):
    '''Convert 2d velocity to rectified RLUD velocity.'''
    out = np.array([
        relu( vels[..., 0]), # right
        relu(-vels[..., 0]), # left
        relu( vels[..., 1]), # up
        relu(-vels[..., 1]), # down
    ]).T
    return out
def wrap_degrees(angle):
    angle = angle % 360
    return np.minimum(angle, 360 - angle)
def inferred_state_refit(curs_pos, tgt_pos, alpha=0.2, theta_threshold=22.5, **kwargs):
    out = np.zeros(7)
    out[-1] = 1.0
    out[0:2] = curs_pos[0:2]
    displacement = tgt_pos - curs_pos
    displacement_l1 = np.sum(np.abs(displacement))
    displacement_angle = np.arctan2(displacement[1], displacement[0]) # Angle relative to xy plane
    displacement_angle_degrees = displacement_angle*180.0/np.pi
    # minimum angle (non-negative) from each of 4 directions right, left, up, down
    degrees_from_RLUD = wrap_degrees((displacement_angle_degrees - np.array([0.0, 180.0, 90.0, 270.0])) % 360)
    if degrees_from_RLUD.min() <= theta_threshold:
        # Snap inferred velocity if the angle to the target is less than theta_threshold
        out[2:6] = (degrees_from_RLUD <= theta_threshold) # one-hot of closest direction angle
    else:
        # vector pointing toward target with L1 norm = 1.
        out[2:6] = rectify_vels(displacement/np.maximum(displacement_l1, 1e-8))
    scale = np.minimum(displacement_l1/alpha, 1.0) # scale to apply to one-hot or simplex velocity
    out[2:6] = scale*out[2:6]
    return out
def inferred_state_refit(curs_pos, tgt_pos, alpha=0.2, theta_threshold=22.5, **kwargs):
    # curs_pos: shape (N, 2) or (2,)
    # tgt_pos: shape (N, 2) or (2,)
    out = np.zeros(curs_pos.shape[0:-1] + (7,))
    out[..., -1] = 1.0
    out[..., 0:2] = curs_pos[..., 0:2]
    displacement = tgt_pos - curs_pos
    displacement_l1 = np.sum(np.abs(displacement), axis=-1, keepdims=True)
    displacement_angle = np.arctan2(displacement[..., 1], displacement[..., 0]) # Angle relative to xy plane
    displacement_angle_degrees = displacement_angle*180.0/np.pi
    scale = np.minimum(displacement_l1/alpha, 1.0) # scale to apply to one-hot or simplex velocity
    rlud_angles = np.array([0.0, 180.0, 90.0, 270.0])
    if curs_pos.ndim == 1:
        degrees_from_RLUD = wrap_degrees((displacement_angle_degrees - rlud_angles) % 360)
    else:
        degrees_from_RLUD = wrap_degrees((displacement_angle_degrees[:, None] - rlud_angles[None, :]) % 360)
    truth = (degrees_from_RLUD <= theta_threshold)
    tmp = (~truth.any(axis=-1, keepdims=True))*rectify_vels(displacement/np.maximum(displacement_l1, 1e-8))
    out[..., 2:6] = truth + tmp # one-hot of closest direction angle
    out[..., 2:6] = scale*out[..., 2:6]
    return out

import ast
def load_data(filename, return_dict=True, copy_arr=False):
    with open(filename, 'rb') as openfile:
        name = openfile.readline().decode('utf-8').strip()
        labels = openfile.readline().decode('utf-8').strip()
        dtypes = openfile.readline().decode('utf-8').strip()
        shapes = None
        # shapes can be indicated with a $ to separate.
        if len(dtypes.split('$')) == 2:
            dtypes, shapes = dtypes.split('$')
            dtypes = dtypes.strip()
            shapes = ast.literal_eval(shapes.strip())
        
        labels = labels.split(',')
        dtypes = dtypes.split(',')
        if shapes is None:
            data = np.fromfile(openfile, dtype=[item for item in zip(labels, dtypes)])
        else:
            data = np.fromfile(openfile, dtype=[item for item in zip(labels, dtypes, shapes)])
        if not return_dict:
            return data
        if copy_arr:
            # copy separates the individual arrays from the bulk numpy array, allowing for memory consolidation.
            data_dict = {label: data[label].copy() for label in labels}
        else:
            data_dict = {label: data[label] for label in labels}
        data_dict['name'] = name
        data_dict['labels'] = labels
        data_dict['dtypes'] = dtypes
    return data_dict

def kf_clda_from_task_data(task_data, kf_init_path=None, offset=0, dt=50000, half_life=1200.0, A_gain=1.0):
    state_task = task_data['state_task'].flatten()
    allow_kf_sync = task_data['allow_kf_sync'].flatten() 
    allow_kf_adapt = task_data['allow_kf_adapt'].flatten()*(~allow_kf_sync)
    
    adapt_ticks = np.nonzero(allow_kf_adapt)[0]
    adapt_ticks = adapt_ticks[adapt_ticks > offset]
    decoded_pos_adapt = task_data['decoded_pos'][adapt_ticks - offset]
    target_pos_adapt = task_data['target_pos'][adapt_ticks - offset]
    decoder_h_adapt = task_data['decoder_h'][adapt_ticks + 1] #logged value corresponds to the next tick for kf_clda module, therefore offset decoder_h by 1 tick
    
    inf_state = inferred_state_refit(decoded_pos_adapt, target_pos_adapt)
    kf_np = np.load(kf_init_path)
    
    # hyperparameters
    dt_s = dt/1e6
    lam = np.exp(np.log(0.5)/(half_life/dt_s))
    kf_model = KalmanFilter(dt=dt_s, lam=lam)
    try:
        kf_model.load(kf_np)
    except Exception as e:
        print(e)
        print('Loading kf failed. Using untrained KF instead.')
    kf_model.A = kf_model.get_A(dt_s, gain=A_gain)
    
    for inf_state_i, decoder_h_i in zip(inf_state, decoder_h_adapt):
        kf_model.process_state_obs(inf_state_i, decoder_h_i, iterate_inv=True, kf_iter=1)
    kf_model.update_M1M2()
    return kf_model

import argparse
import os
import pathlib
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_data_path', type=str, help='task_data path to .bin file.')
    parser.add_argument('--kf_init_path', type=str, help='Path to initial kf .npz file.')
    parser.add_argument('--dt', type=float, default=50000, help='dt in microseconds.')
    parser.add_argument('--half_life', type=float, default=1200.0, help='half_life, in seconds.')
    parser.add_argument('--offset_s', type=float, default=0.0, help='offset, in seconds.')
    parser.add_argument('--A_gain', type=float, default=0.5, help='A_gain for velocity to position elements.')
    parser.add_argument('--kf_save_path', type=str, help='Where to safe kf .npz file afterward.')
    args = parser.parse_args()
    
    if args.kf_save_path is None:
        raise ValueError('kf_save_path not provided!')
    kf_init_path = args.kf_init_path
    if kf_init_path is None:
        kf_init_path = pathlib.Path(args.task_data_path).parents[0] / 'init_kf.npz'
    
    offset = int(args.offset_s/(args.dt/1e6))
    task_data = load_data(args.task_data_path)
    kf_model = kf_clda_from_task_data(task_data, kf_init_path=kf_init_path, offset=offset, dt=args.dt, half_life=args.half_life, A_gain=args.A_gain)
    kf_model.save(path=args.kf_save_path)
    pass