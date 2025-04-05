import numpy as np
import socket
from modules.connection_manager import ConnectionManager
'''
# in modules
  send_kf_vel:
    constructor: True
    destructor: True
    sync:
      - kf_clda
    in:
      - kf_state
    trigger: False
    params:
      mode: client # 'server' or 'client'
      IP: '123.456.789.000'
      PORT: 7781?
      rotation_degrees: 0.0
      flipx: False
'''

mode = params.get('mode', 'server') # 'server' or 'client'
IP = params.get('IP', socket.gethostbyname(socket.gethostname()))
PORT = params['PORT']
rotation_degrees = params.get('rotation_degrees', 0.0)
flipx = params.get('flipx', False)
flipy = params.get('flipy', False)

rotation_radians = rotation_degrees/180.0*np.pi
rotation_matrix = np.array([[np.cos(rotation_radians), -np.sin(rotation_radians)], [np.sin(rotation_radians), np.cos(rotation_radians)]])

def kf_state_to_vel(kf_st, normalize='l1', clip=False, alpha=0.2):
    '''
    assumes kf_st is [posx, posy, vel+x, vel-x, vel+y, vel-y, 1]
    normalize: {'l1', 'l2', 'linf', None}
    velocities with norm greater than or equal to alpha are mapped to a norm of 1.
      less than that, the norm is scaled linearly to zero.
    '''
    if clip:
        kf_st[2:6] = np.clip(kf_st)
    vel = np.array([kf_st[2] - kf_st[3], kf_st[4] - kf_st[5]])

    if normalize == 'l1':
        l1 = np.abs(vel).sum()
        scale = np.minimum(l1/alpha, 1.0)
        vel = scale*vel
    elif normalize == 'l2':
        l2 = np.sqrt((vel**2).sum())
        scale = np.minimum(l2/alpha, 1.0)
        vel = scale*vel
    elif normalize == 'linf':
        # equivalent to projection along the largest cardinal direction.
        amin = np.argmin(np.abs(vel))
        amax = np.argmax(np.abs(vel))
        vel[amin] = 0
        linf = np.abs(vel[amax])
        scale = np.minimum(linf/alpha, 1.0)
        vel = scale*vel
    elif normalize == None:
        pass
    else:
        raise ValueError('')
    
    return vel

conn = ConnectionManager(ip=IP, port=PORT, mode=mode, shape=(2,))