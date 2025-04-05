import numpy as np
import ast

'''
# example:
  replay:
    constructor: true
    destructor: false
    sync:
      - logger
    trigger: true
    out:
      - eegbuffersignal
      - databuffer
      - eegbufferindex
      - state_task
      - decoder_output
      - decoded_pos
      - cursor_pos
      - target_pos
      - target_size
      - game_state
      - kf_state
      - kf_update_flag
      - kf_C
      - allow_kf_adapt
      - allow_kf_sync
    params:
      replay_data_folder: /data/raspy/2023-01-01_Subject_session_1/
      log:
        task:
          # these are the signals to load every tick
          signals:
            - state_task
            - decoder_output
            - decoded_pos
            - cursor_pos
            - target_pos
            - target_size
            - game_state
            - allow_kf_adapt
            - allow_kf_sync
        eeg:
          # these buffers are loaded based on their alignment to the align_stream data each tick.
          index_out: eegbufferindex
          align_stream: task # which stream to align the buffers to
          buffers:
            - eegbuffersignal
            - databuffer
'''

replay_data_folder = params['replay_data_folder']
# this is align_stream's step for buffers. Use a separate module instance to account for multiple streams.
starting_step = params['starting_step'] if 'starting_step' in params else 0 # not recommended because I think it's bugged.
log = params['log'] # defines the logging stream names and variables
ignore_end = params.get('ignore_end', False)
for stream_name in log.keys():
    log[stream_name]['step'] = starting_step

def load_data(filename, return_dict = True):
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
        data_dict = {label: data[label] for label in labels}
        data_dict['name'] = name
        data_dict['labels'] = labels
        data_dict['dtypes'] = dtypes
    return data_dict

data = {}
for stream_name in log.keys():
    fname = replay_data_folder + stream_name + '.bin'
    if 'signals' in log[stream_name]:
        data[stream_name] = load_data(fname)
    if 'buffers' in log[stream_name]:
        align_stream = log[stream_name]['align_stream']
        align_stream_data = load_data(replay_data_folder + align_stream + '.bin')
        data[stream_name] = load_data(fname)
        data[stream_name]['align_idx'] = align_stream_data[stream_name + '_step'].copy() # Array of indices of this stream to access.
        del align_stream_data
        data[stream_name]['idx_'] = data[stream_name]['align_idx'][log[stream_name]['step']] + 1 # index of this stream to access next.
        data[stream_name]['N'] = globals()[log[stream_name]['buffers'][0]].shape[0] // 2 # Half the size of the bipartite buffer
        data[stream_name]['index_out'] = globals()[log[stream_name]['index_out']]
        data[stream_name]['index_out'][:] = (data[stream_name]['idx_'] % data[stream_name]['N']) + data[stream_name]['N']