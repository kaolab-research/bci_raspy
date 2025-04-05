tt = time.time_ns() # current time in ns
if pause_condition == 'all':
    condition = np.all(internal_pause_state)
else:
    # i.e. if pause_condition == 'any'
    condition = np.any(internal_pause_state)
if not condition:
    for stream_name in log.keys():
        # For buffers, get the number of new records (dstep)
        if 'buffers' in log[stream_name]:
            log[stream_name]['new_idx'] = globals()[log[stream_name]['index']][0]
            dstep = (log[stream_name]['new_idx'] - log[stream_name]['idx']) % log[stream_name]['N']
            log[stream_name]['step'] += dstep # Step is the most recent total number of records for this stream.
            log[stream_name]['dstep'] = dstep
        elif 'records' in log[stream_name]:
            dstep = int(globals()[log[stream_name]['flag']][0]) # bool to int
            log[stream_name]['step'] += dstep # Step is the most recent total number of records for this stream.
            log[stream_name]['dstep'] = dstep
        else:
            dstep = 1
            log[stream_name]['step'] += dstep # Step is the most recent total number of records for this stream.
            log[stream_name]['dstep'] = dstep
    for stream_name in log.keys():
        # Send new buffer records one record at a time.
        # Buffers are bipartite buffers where idx is the index that the next entry will be recorded. Let idx >= N
        # Number of records of the buffers is indicated by the change in the index.
        # Warning: will be slow for large number of new records!
        # Possible to-do: switch to contiguous block saving for faster processing.
        
        if 'buffers' in log[stream_name]:
            idx = log[stream_name]['idx'] # index to begin reading from
            N = log[stream_name]['N']
            n_samples = log[stream_name]['dstep']
            
            data_dump_list = []
            for vname in log[stream_name]['buffers']:
                v = globals()[vname]
                data_dump_list.append(v[idx-N:idx-N+n_samples])
            for other_name in log[stream_name]['other_names']:
                data_dump_list.append(np.full(n_samples, log[other_name]['step']))
            data_dump_list.append(np.full(n_samples, tt))
            data_to_dump = np.array(list(zip(*data_dump_list)), dtype=log[stream_name]['dtype'])
            dump_bytes = data_to_dump.tobytes()
            for connection_name in connections.keys():
                client = connections[connection_name][stream_name]['client']
                client.sendall(dump_bytes)
            idx = (idx - N + n_samples) % N + N
            log[stream_name]['idx'] = log[stream_name]['new_idx']
        
        # Send the signals each time this loop is called.
        if 'signals' in log[stream_name]:
            data_dump_list = []
            for vname in log[stream_name]['signals']:
                v = globals()[vname]
                data_dump_list.append(v)
            for other_name in log[stream_name]['other_names']:
                data_dump_list.append(log[other_name]['step'])
            data_dump_list.append(tt)
            data_to_dump = np.array(tuple(data_dump_list), dtype=log[stream_name]['dtype'])
            dump_bytes = data_to_dump.tobytes()
            for connection_name in connections.keys():
                client = connections[connection_name][stream_name]['client']
                client.sendall(dump_bytes)
        
        # For pickled records. Sends as pickled dictionary containing the records and the step number values for synchronization
        if 'records' in log[stream_name]:
            flag = globals()[log[stream_name]['flag']][0]
            if flag:
                vs = [globals()[vname] for vname in log[stream_name]['records']] # record variables
                ls = [int(globals()[vname]) for vname in log[stream_name]['lengths']] # record variable lengths

                # full_record is {variable0: value0, variable1: value1, ..., other0_step: other0_step_no, other1_step: other1_step_no, ...}
                full_record = {
                    **{vname: pickle.loads(v[0:l]) for v, l, vname in zip(vs, ls, log[stream_name]['records'])},
                    **{other_name + '_step': log[other_name]['step'] for other_name in log[stream_name]['other_names']}
                }

                dump_bytes = pickle.dumps(full_record)
                for connection_name in connections.keys():
                    client = connections[connection_name][stream_name]['client']
                    client.sendall(dump_bytes)

if pause_var is not None:
    internal_pause_state[:] = pause_var[:]