for connection_name in connections.keys():
    for stream_name in connections[connection_name]:
        client.sendall(b'')