if conn is None:
    try:
        conn, addr = server.accept()
        conn.setblocking(False)
        print("Gaze connection established!")
    except:
        pass
else:
    try:
        buffer += conn.recv(2**30)
    except Exception as e:
        pass


# e.g. dtype is float64, shape (2,) -> 16 bytes
n_samples = len(buffer) // record_num_bytes
for n in range(n_samples):
    gaze_buffer[gaze_buffer_idx] = gaze_buffer[gaze_buffer_idx-bufferoffset] = np.frombuffer(buffer[0:record_num_bytes])
    buffer = buffer[record_num_bytes:None]
    gaze_buffer_idx[:] = ((gaze_buffer_idx - bufferoffset) + 1) % bufferoffset + bufferoffset

