
import numpy as np
import os
from modules.kf_util import KalmanFilter

'''
signals:
  kf_state:
    shape: (7,)
    dtype: float64
  decoder_h:
    shape: (>=hdim,)
    dtype: float32
  kalman_gain:
    shape: (7, >=hdim+1)
    dtype: float64
  rluds_output:
    shape: (5,)
    dtype: float32
'''

dt = params['dt'] # in microseconds
dt_s = dt/1e6 # in seconds
init_EBS_seconds = params['init_EBS_seconds'] # Effective batch size to assign to training sufficient statistics, in seconds (corresponding to infinite half-life)
half_life = params['half_life'] # half-life of sufficient statistics, in seconds    
kf_init_path = params['kf_init_path'] if 'kf_init_path' in params else None # string.
RLUDs_idx = params['RLUDs_idx'] # indices corresponding to right, left, up, down, resepctively
continuous_update = params['continuous_update'] if 'continuous_update' in params else False # whether to continuously update params
A_gain = params['A_gain'] if 'A_gain' in params else 1.0 # scaling the components that are +/- dt by default.
inferred_state_mode = params['inferred_state_mode'] if 'inferred_state_mode' in params else 'refit'
refit_mode = params.get('refit_mode', 'vector') # vector or split
obs_name = params.get('obs_name', 'decoder_h')
obs_arr = globals()[obs_name]
print('warning: refit_mode', f"'{refit_mode}'")
# inferred_state_mode of 'refit' will use decoded_pos and target_pos to calculate inferred state
# if inferred_state_mode is a dictionary, then the key corresponds to the state_task,
#   and the value corresponds to 'left', 'right', 'up', 'down'
# e.g.
'''
    params:
      inferred_state_mode: refit
# or 
    params:
      ...
      ...
      ...
      inferred_state_mode:
        0: left
        1: right
        2: up
        3: down
        10: right
'''

if half_life in ['inf', 'infinite', 'infinity']:
    lam = 1.0
else:
    # lam**(half_life/dt_s) = 0.5
    lam = np.exp(np.log(0.5)/(half_life/dt_s))

if init_EBS_seconds in ['steadystate', 'steady-state', 'steady_state']:
    if lam < 1.0:
        init_EBS = 1/(1 - lam)
    else:
        raise ValueError(f'init_EBS_seconds {init_EBS_seconds} is incompatible with lambda of {lam} corresponding to half_life {half_life}')
elif init_EBS_seconds in ['prev', 'previous']:
    init_EBS = None
else:
    init_EBS = KalmanFilter.get_ebs(init_EBS_seconds/dt_s, lam) # WHICH ONE??

print(f'Warning: using initial EBS of {init_EBS} corresponding to {init_EBS_seconds} seconds')

if continuous_update:
    print('warning: continuous update ON')
    assert (continuous_update is True)
    print('warning: continuous update is still ON')
else:
    print('warning: continuous update OFF')
    assert (continuous_update is False)
    print('warning: continuous update is still OFF')

def gen_default_state():
    out = np.zeros(7)
    out[-1] = 1.0
    return out
def relu(x):
    return x*(x > 0)
def rectify_vels(vels):
    '''Convert 2d velocity to rectified RLUD velocity.'''
    out = np.array([
        relu( vels[0]), # right
        relu(-vels[0]), # left
        relu( vels[1]), # up
        relu(-vels[1]), # down
    ])
    return out
def inferred_state_cache(kf_state, target_pos, decoded_vel=None, alpha=0.2, theta_threshold=10.0, **kwargs):
    # generates the inferred kalman filter state
    # where the combined velocity components point from kf_state[0:2] (position) to target_pos.
    # If decoded_vel is not None, instead combine decoded_vel and the
    #   optimal direction, weighting the former as min(aligned_l1/scale, 1.0)

    # kf_state: shape 7 of [xpos, ypos, +xvel, -xvel, +yvel, -yvel, 1]
    # target_pos: (2,)
    # decoded_vel: (2,), np.ndarray
    # alpha: distance at which to start attenuating velocity
    # theta_threshold: degrees
    theta_threshold_radians = theta_threshold*np.pi/180.0 # unused

    # sum of absolute value of target velocity scales from 0 to 1 between
    #   0 <= sum(abs(displacement)) < alpha, and is 1 for larger displacements.
    out = np.zeros(7)
    out[-1] = 1.0
    out[0:2] = kf_state[0:2]

    displacement = target_pos - kf_state[0:2]
    displacement_l1 = np.sum(np.abs(displacement)) # l1 norm
    scale = np.minimum(displacement_l1/alpha, 1.0) # overall scale.
    normalized_displacement = displacement/np.maximum(np.sum(np.abs(displacement)), 1e-8) # displacement with l1 norm equaling 1.
    rectified_displacement = rectify_vels(normalized_displacement) # 4d RLUD normalized displacement
    if decoded_vel is None:
        out[2:6] = scale*rectified_displacement # scale rectified_displacement if displacement_l1 < alpha. else scale = 1.0
    else:
        # # align decoded_vel to displacement, i.e. only keep components that are pointing in the same direction
        aligned_vel = (np.sign(decoded_vel) == np.sign(displacement))*decoded_vel
        aligned_l1 = np.sum(np.abs(aligned_vel))
        aligned_vel_weight = np.minimum(aligned_l1/np.maximum(scale, 1e-8), 1.0) # weight to assign to aligned_vel
        # out[2:6] is a scaled convex combination of rectified_displacement and rectify_vels(aligned_vel)
        out[2:6] = scale*((1 - aligned_vel_weight)*rectified_displacement + aligned_vel_weight*rectify_vels(aligned_vel))
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

class InferredStateByDict():
    def __init__(self, inferred_state_map):
        self.st_task_to_direction = {int(key): value for (key, value) in inferred_state_map.items()}
        return
    def __call__(self, curs_pos, tgt_pos, st_task=None, alpha=0.2, theta_threshold=22.5, **kwargs):
        out = np.zeros(7)
        out[-1] = 1.0
        out[0:2] = curs_pos[0:2]
        st_task = int(st_task)
        val = self.st_task_to_direction.get(st_task, None)
        if val is None:
            return None
        scale = 1.0
        if val in ['right', 'Right', 'R']:
            out[2] = 1.0
        elif val in ['left', 'Left', 'L']:
            out[3] = 1.0
        elif val in ['up', 'Up', 'U']:
            out[4] = 1.0
        elif val in ['down', 'Down', 'D']:
            out[5] = 1.0
        out[2:6] = scale*out[2:6]
        return out



###### Chooose which method to infer state from.
if inferred_state_mode == 'refit':
    gen_inferred_state = inferred_state_refit
elif isinstance(inferred_state_mode, dict):
    gen_inferred_state = InferredStateByDict(inferred_state_mode)
else:
    raise ValueError(f'invalid inferred_state_mode of {inferred_state_mode}')

# kf_state should have shape (7,), i.e.
# [xpos, ypos, right, left, up, down, 1]
kf_state[:] = gen_default_state()
# the other signal is kf_obs, which should have shape (obs_dim,)


# effective obs_dim is changed after loading.
kf_model = KalmanFilter(state_dim=7, obs_dim=2, dt=dt_s, lam=lam)
try:
    kf_init_np = np.load(kf_init_path)
    kf_model.load(kf_init_np) # loads R, S, T(, Tinv), EBS, S_k, K_k, A, W. Sets C, Q, and Qinv from [R, S, T]
    kf_model.A = kf_model.get_A(dt_s, gain=A_gain)
    kf_model.rescale_EBS(init_EBS)
    print('Kalman Filter model loaded!')
except:
    raise NotImplementedError('kf must be initialized using kf_init_path.')

kf_model.update_M1M2(verbosity=1) # Generate recursion variables M1 and M2 from K_k, C, A
kf_model.save(path=os.path.abspath(params['data_folder']) + '/init_kf.npz')

obs_dim = kf_model.T.shape[0]

from collections import deque
inf_state_queue = deque([])
inf_state_delay = params.get('inf_state_delay', 0.0)
queue_length = int(inf_state_delay/dt_s)
print('warning: queue_length', queue_length)