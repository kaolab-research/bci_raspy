# optional.
resave_npz = params['resave_npz'] if 'resave_npz' in params else False # not compatible with replay module.
delete_bin = params['delete_bin'] if 'delete_bin' in params else False
connections = params['connections']

from modules.data_util import load_data

if resave_npz:
    print('resaving .bin files as .npz files...')
    for connection_name in connections:
        for stream_name in connections[connection_name]:
            filename = connections[connection_name][stream_name]['filename']
            data = load_data(filename)
            np.savez_compressed(filename.replace('.bin', '.npz'), **data)
            if delete_bin:
                os.remove(filename)