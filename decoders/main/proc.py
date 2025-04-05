# import argparse # Not used currently. May be useful.
import multiprocessing as mp
import multiprocessing.shared_memory
import util
import numpy as np
import traceback
import sys
import platform

def proc(signals, module, p_in: list, p_out: list, trigger=None, log_queue=None,
        timeout=60.0, loop_timeout=10.0):
    # signals: dictionary containing information about the signals
    # module: dictionary containing information about the current module
    # p_in and p_out: Connection objects (created using mp.Pipe). Only for sync'ed processes
    # trigger: pipe to main if not None
    # log_queue mp.Queue object to add print output to.
    # timeout: time to wait for upstream messages for the first&second loop. Needs to be longer than expected dt.
    # loop_timeout: time to wait for upstream messages for subsequent loops. Needs to be longer than expected dt.
    
    data_folder = module['params']['data_folder']
    if log_queue is not None and platform.system() == 'Windows': # for Windows only. Due to fork/spawn differences
        log_filename = data_folder + 'logfile.log'
        # class which captures print output. Writes to terminal and puts to log_queue
        class Logger(object):
            def __init__(self):
                self.terminal = sys.stdout # Get and save (a reference to) the original stdout/print stream.

            def write(self, message):
                self.terminal.write(message) # Actually print the stdout message.
                log_queue.put(message) # Add the message to the queue.

            def flush(self):
                #this flush method is needed for python 3 compatibility.
                #this handles the flush command by doing nothing.
                #you might want to specify some extra behavior here.
                pass
        sys.stdout = Logger()
    
    print('  starting {}'.format(module['name']))
    
    d_ = {} # Namespace dictionary containing all the objects created and used
    m_shm = util.shm_setup(signals, module) # Shared memory setup
    for key, value in m_shm.items():
        if key[0] != '_':
            d_[key] = value # Assign numpy arrays to namespace d_
    d_['params'] = module['params'] if ('params' in module) else {}
    
    if 'constructor' in module: # Execute constructor (if required) in namespace d_
        if module['constructor']:
            constructor_fname = module['path'] + '_constructor.py'
            with open(constructor_fname, 'r') as f:
                constructor = f.read()
                print('before constructor ' + module['name'])
                exec(constructor, d_)
                print('after constructor ' + module['name'])
    destructor = None
    if 'destructor' in module: # Get destructor (if required) for future execution
        if module['destructor']:
            destructor_fname = module['path'] + '_destructor.py'
            with open(destructor_fname, 'r') as f:
                destructor = f.read()
                destructor = compile(destructor, destructor_fname, 'exec')
    fname = module['path'] + '.py'
    
    use_loop = module['loop'] if 'loop' in module else True # Whether this modules uses a loop.
    if use_loop:
        with open(fname, 'r') as f:
            loop = compile(f.read(), fname, 'exec') # Get and compile looped code
    
    # Receive the trigger signal from main before signal (if applicable)
    msg = ''
    if trigger is not None:
        if not trigger.poll(timeout):
            pass # return?
        else:
            msg = trigger.recv()
    
    i = 0 # Counter of no. of loops completed. Empty cycles count as a loop
    cycle = module['cycle'] if 'cycle' in module else 1 # Number of cycles to wait from upstream modules for each execution.
                                                        # 1 means execute on every cycle. 
                                                        # Caution: Any value other than 1 probably messes with timing.
    print('before loop', module['name']) # Just to keep track of where we are in execution.
    if use_loop:
        while True:
            try:
                # Get a message from main via trigger pipe if it's there
                if trigger is not None:
                    if trigger.poll(0.):
                        msg = trigger.recv()
                    if msg == 'quit_':
                        raise Exception('quit by trigger')
                if i > 1:
                    # Initial loading/first&second loop usually takes a lot longer. Switch to smaller loop timeout here.
                    timeout = loop_timeout
                # Poll the upstream pipes.
                if i > 0 or trigger is None:
                    for p in p_in:
                        if not p.poll(timeout):
                            raise Exception('poll timed out')
                        else:
                            msg = p.recv()
                            if msg == 'quit_':
                                raise Exception('quit by message')
                # Only run the loop every cycle number of loops.
                if i % cycle == 0:
                    if not any([p.poll() for p in p_in]): # Prevents this process from playing catch-up.
                        # p.poll() should return immediately.
                        exec(loop, d_)
                # You can initiate a quit from within a module if you create a variable quit_ which evaluates to True.
                if 'quit_' in d_.keys():
                    if d_['quit_']:
                        raise Exception('quit by variable')
                # Send a message to the downstream pipes.
                for p in p_out:
                    p.send('')
                i += 1
            except (Exception, KeyboardInterrupt) as e:
                e_str = 'Exception: '
                if str(e) in ['', 'quit', 'poll timed out', 'quit by trigger', 'quit by message', 'quit by variable']:
                    e_str += str(e)
                else:
                    e_str += traceback.format_exc()
                print('!!! ' + module['name'] + ' ' + e_str + '\n', end='') # Print the exception.
                
                # Tell downstream modules and main module to quit.
                for p in p_out:
                    try:
                        p.send('quit_')
                    except:
                        pass
                if trigger is not None:
                    trigger.send('quit_')
                break
    
    # Execute the destructor (if applicable)
    if destructor is not None:
        print('    Destructing {}\n'.format(module['name']), end='')
        exec(destructor, d_) # Execute destructor code
    for key, value in m_shm.items():
        if key[0] == '_':
            m_shm[key].close() # Close this process's access to shared memory
    
    print('  Exiting {}\n'.format(module['name']), end='') # Print a status update.
    return
