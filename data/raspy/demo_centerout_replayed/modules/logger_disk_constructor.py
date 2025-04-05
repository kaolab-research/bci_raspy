import numpy as np
import socket
import select
import time
import datetime
import os
import ast
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import contextlib
import argparse

def logger_func(data_folder, connections):
    print('opening logger_disk ports')
    with contextlib.ExitStack() as stack:
        for connection_name in connections: # should probably only have one item. Otherwise need to modify filename
            for stream_name in connections[connection_name]:
                IP = connections[connection_name][stream_name]['IP']
                PORT = connections[connection_name][stream_name]['PORT']
                server = stack.enter_context(socket.socket(family = socket.AF_INET, type = socket.SOCK_STREAM))
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Set server to be able to reuse the address.
                server.bind((IP, PORT))
                server.listen(1)
                connections[connection_name][stream_name]['server'] = server
                
                filename = data_folder + stream_name + '.bin'
                connections[connection_name][stream_name]['filename'] = filename
                connections[connection_name][stream_name]['file'] = stack.enter_context(open(filename, 'wb'))
                connections[connection_name][stream_name]['nbytes'] = 0
                connections[connection_name][stream_name]['header_len'] = -1
                connections[connection_name][stream_name]['header'] = bytes(0)

        for connection_name in connections:
            for stream_name in connections[connection_name]:
                server = connections[connection_name][stream_name]['server']
                conn, addr = server.accept()
                connections[connection_name][stream_name]['conn'] = conn
                connections[connection_name][stream_name]['addr'] = addr
        # Set all connections to non-blocking.
        for connection_name in connections:
            for stream_name in connections[connection_name]:
                conn = connections[connection_name][stream_name]['conn']
                conn.setblocking(False)

        time.sleep(0.01)
        quit = False
        while True:
            try:
                for connection_name in connections:
                    for stream_name in connections[connection_name]:
                        conn = connections[connection_name][stream_name]['conn']
                        try:
                            msg = conn.recv(2**30) # Receive the message.
                            # If the header length hasn't been set yet:
                            if connections[connection_name][stream_name]['header_len'] == -1:
                                # What happens if you only receive one bytes? I don't know. 
                                # Hopefully it doesn't happen
                                if len(msg) == 1:
                                    print('\nPLEASE fix me\n')
                                connections[connection_name][stream_name]['header_len'] = int.from_bytes(msg[0:2], 'little')
                                msg = msg[2:]
                            header_len = connections[connection_name][stream_name]['header_len'] # Number of bytes of the header.
                            ll = len(connections[connection_name][stream_name]['header'])
                            if ll < header_len:
                                jj = min(header_len-ll, len(msg))
                                connections[connection_name][stream_name]['header'] += msg[0:jj]
                            f = connections[connection_name][stream_name]['file']
                            f.write(msg) # Write the msg to the file without any processing
                            connections[connection_name][stream_name]['nbytes'] += len(msg) # Increment no. of bytes.
                            if msg == b'':
                                quit = True
                        except OSError:
                            # catch non-blocking exceptions.
                            pass
                if quit:
                    raise Exception('client closed connection')
                time.sleep(0.01)
            except (KeyboardInterrupt, Exception) as e:
                print('logger_disk Exception:', e)
                break
        # Close files and connections.
        #for connection_name in connections:
        #    for stream_name in connections[connection_name]:
        #        connections[connection_name][stream_name]['file'].close()
        #        connections[connection_name][stream_name]['conn'].close()
        print('####logger_disk: saved to', data_folder)
        for connection_name in connections:
            for stream_name in connections[connection_name]:
                print('    ' + stream_name, ':', connections[connection_name][stream_name]['nbytes'], 'bytes', ','
                    'header_len', connections[connection_name][stream_name]['header_len'], ',\n'
                    'header:\n' + connections[connection_name][stream_name]['header'].decode())
    return

if __name__ in ['builtins']:
    # __name__ will be 'builtins' when run using RASPy
    connections = params['connections'] # Dictionary specifying the connections to be made.
    data_folder = params['data_folder'] # directory to save .bin files (and others, for other modules) within.
    logger_func(data_folder, connections)
elif __name__ in ['__main__']:
    # For standalone execution.
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', help='directory to save .bin files in')
    parser.add_argument('yaml_path', help='path to .yaml file specifying the connections. Similar to raspy but connections is at top of hierarchy.')
    args = parser.parse_args()

    data_folder = os.path.join(args.data_folder, '')
    with open(args.yaml_path, 'r') as yaml_file:
        yaml_data = yaml.load(yaml_file, Loader=Loader)
        connections = yaml_data['connections']
    logger_func(data_folder, connections)