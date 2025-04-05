import numpy as np
import multiprocessing as mp
import multiprocessing.shared_memory
import time
import socket

def shm_setup(signals, module):
    # signals: dict containing information for shared memory variables. From YAML.
    # module: dict containg information for relevant module. From YAML.
    # Returns: a dictionary containing numpy arrays which use shared memory buffers.
    shm_dict = {}
    module_signals = [] # list/set of relevant signals for the inputted module.
    if 'in' in module: # maybe replace 'in', and 'out' with something else.
        module_signals += module['in']
    if 'out' in module:
        module_signals += module['out']
    module_signals = list(set(module_signals))
    
    for name in module_signals:
        signal = signals[name] # info for the current signal
        shm = shm_dict['_' + name] = mp.shared_memory.SharedMemory(name=name) # Shared memory buffer
        shm_dict[name] = np.ndarray(signal['shape'], dtype=signal['dtype'], buffer=shm.buf)
            # Numpy array corresponding to shared memory buffer
    return shm_dict

def verify_trigger(modules):
    # Returns True if module directed graph is good.
    # Returns False if module directed graph is bad or there's a bug.
    for name in modules.keys():
        if not verify_trigger_module(modules, name):
            return False
    return True

def verify_trigger_module(modules, name, origin=None):
    # Returns True if the cyclic graph containing this module (as checked by flowing downstream)
    #     contains a module where 'trigger' is True 
    if 'trigger' in modules[name]:
        if modules[name]['trigger']:
            return True
    if origin == name:
        return False
    else:
        if modules[name]['sync'] is not None:
            if origin is None:
                origin = name
            if False in [verify_trigger_module(modules, name2, origin) for name2 in modules[name]['sync']]:
                return False
    return True

# probably defunct. See /modules/logger_disk_constructor.py instead
'''
def logger_func(filename, IP, PORT, timeout=10.0):
    serverSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_STREAM)
    serverSocket.bind((IP, PORT))
    serverSocket.listen(1)
    conn, addr = serverSocket.accept()
    
    formatMessage = conn.recv(4096)
    fo = open(filename, 'wb')
    fo.write(formatMessage)
    #print(formatMessage.decode('utf-8'))
    started = False
    while True:
        try:
            #print('i', i)
            #if not conn.poll(timeout):
            #    raise Exception('poll timed out')
            msg = conn.recv(2**31)
            #print(msg)
            #if msg == 'quit':
            #    raise Exception('quit')
            #print(len(msg))
            if len(msg) == 0 and started: # Maybe use a different checker
                raise Exception('no bytes received')
            elif len(msg) > 0 and not started:
                started = True
            fo.write(msg)
            #print
            time.sleep(0.01)
        except (KeyboardInterrupt, Exception) as e:
            print('logger_func Exception:', e)
            break
    conn.close()
    fo.close()
    print('exiting logger_func. Saved to', filename)
    return
'''