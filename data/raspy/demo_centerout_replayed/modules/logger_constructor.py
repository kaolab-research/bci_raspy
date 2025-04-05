import os
import datetime
import time
import socket
import numpy as np
import multiprocessing as mp
import util # imports from main folder. Maybe change.
import pickle

#Expected params: (needs to be updated)
#  log (a list):
#    - variable_1 # each variable should be single-dimensional
#    - variable_2
#  logeeg:
#    logeeg: bool
#    buffer: eegbuffersignal (name of the buffer) (only if logeeg = True)
#    index: eegbufferindex (only if logeeg = True)
#  logger_type: local # 'local' or 'tcp'
#  save_path: # required if logger_type = local. Should be terminated by '/'
#  IP: # required if logger_type = tcp
#  PORT: # required if logger_type = tcp
#  pause: # (optional): name of boolean pause variable

encoding = 'utf-8'
log = params['log']
connections = params['connections']
try:
    pause_var = globals()[params['pause']] if 'pause' in params else None # True when paused. Best to use bool.
except:
    raise Exception('Variable {} is not available as a signal to this module!'.format(params['pause']))
pause_condition = params['pause_condition'] if 'pause_condition' in params else 'any' # options: 'any' or 'all'
internal_pause_state = np.zeros_like(pause_var) # True when paused. Lags pause_var by 1 tick.

# Setup. 
for stream_name in log.keys():
    log[stream_name]['dtype'] = []
    log[stream_name]['dtype_str'] = ''
    log[stream_name]['labels'] = []
    log[stream_name]['step'] = -1
    log[stream_name]['dstep'] = 0
    log[stream_name]['other_names'] = []
    
    # set up dtype, labels, and whatever else necessary
    if 'signals' in log[stream_name]:
        for i, vname in enumerate(log[stream_name]['signals']):
            v = globals()[vname]
            log[stream_name]['labels'].append(vname)
            log[stream_name]['dtype'].append((vname, v.dtype.str, v.shape))
        
    if 'buffers' in log[stream_name]:
        for i, vname in enumerate(log[stream_name]['buffers']):
            v = globals()[vname]
            log[stream_name]['labels'].append(vname)
            log[stream_name]['dtype'].append((vname, v.dtype.str, v[0].shape))
            if i == 0:
                # N is half the total length of the bipartite buffer.
                log[stream_name]['N'] = globals()[log[stream_name]['buffers'][0]].shape[0] // 2
            
        log[stream_name]['idx'] = globals()[log[stream_name]['index']][0] # should be 1-d, size 1.
                                    # Basically a reference to the variable to be used as the index.
        log[stream_name]['new_idx'] = log[stream_name]['idx']
    # these are saved directly with no header. Load with pickle.
    if 'records' in log[stream_name]:
        log[stream_name]['dtype'] = 'B' # unused but necessary here
        log[stream_name]['labels'] = '' # unused but necessary here
    
    # Add synchronization variables.
    log[stream_name]['other_names'] = [other_name for other_name in log.keys() if other_name != stream_name]
    if 'records' not in log[stream_name]:
        for i, other_name in enumerate(log[stream_name]['other_names']):
            log[stream_name]['dtype'].append((other_name + '_step', '<i4', ()))
            log[stream_name]['labels'].append(other_name + '_step')
        log[stream_name]['dtype'].append(('time_ns', '<i8', ()))
        log[stream_name]['labels'].append('time_ns')

        log[stream_name]['labels'] = ','.join(log[stream_name]['labels'])
        log[stream_name]['dtype_str'] = ','.join([dt[1] for dt in log[stream_name]['dtype']]) + '$' + ','.join([str(dt[2]) for dt in log[stream_name]['dtype']])

        print('  ' + stream_name + ' dtype ' + log[stream_name]['dtype_str'] + ',' + \
              str(np.dtype(log[stream_name]['dtype']).itemsize) + ' bytes' + '\n', end='')
        print('  ' + stream_name + ' labels ' + log[stream_name]['labels'] + '\n', end='')


# Connect to each server and send some information.
for connection_name in connections.keys():
    for stream_name in connections[connection_name]:
        # If ordering matters, I'm sorry.
        time.sleep(0.1) # give other end leeway?
        print('    attempting to connect to:', connection_name, stream_name)
        IP = connections[connection_name][stream_name]['IP']
        PORT = connections[connection_name][stream_name]['PORT']
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((IP, PORT))
        connections[connection_name][stream_name]['client'] = client
        labels = log[stream_name]['labels']
        dtype_str = log[stream_name]['dtype_str']
        header = ''
        if 'records' not in log[stream_name]:
            header += stream_name + '\n'
            header += labels + '\n'
            header += dtype_str + '\n'
        print(stream_name, ' len header ', len(header))
        client.sendall(int.to_bytes(len(header), 2, 'little')) # Send length of header.
        if len(header) > 0:
            client.sendall(header.encode(encoding)) # Send header as single message.
        print('    successfully connected to: {}\n'.format(stream_name), end='')
print('logger constructor end')