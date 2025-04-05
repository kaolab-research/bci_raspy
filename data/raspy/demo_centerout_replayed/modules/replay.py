for stream_name in log.keys():
    # Load in signals (one sample per tick).
    if 'signals' in log[stream_name]:
        for i, vname in enumerate(log[stream_name]['signals']):
            v = globals()[vname]
            step = log[stream_name]['step']
            if step < len(data[stream_name][vname]):
                v[:] = data[stream_name][vname][step]
            else:
                if not ignore_end:
                    quit_ = True
                    print(f"SIGNAL: Replay Reached the end of experiment: {step}")
        log[stream_name]['step'] += 1
    
    # Load in buffers (can have multiple samples per tick, or zero).
    if 'buffers' in log[stream_name]:
        step = log[stream_name]['step'] # Step no. of align_stream
        idx = data[stream_name]['idx_'] # Total samples for this buffer prior to this time
        new_idx = data[stream_name]['align_idx'][step] + 1 # Next sample no. of buffers
        nSamples = new_idx - idx # Number of new samples
        
        N = data[stream_name]['N']
        load_idxs = np.arange(idx, new_idx)
        write_idxs = (load_idxs % N) + N
        for vname in log[stream_name]['buffers']:
            v = globals()[vname]
            v[write_idxs, :] = v[write_idxs - N, :] = data[stream_name][vname][load_idxs]
        log[stream_name]['step'] += 1
        data[stream_name]['idx_'] = new_idx
        data[stream_name]['index_out'][:] = ((data[stream_name]['index_out'] + nSamples) % data[stream_name]['N']) + data[stream_name]['N']