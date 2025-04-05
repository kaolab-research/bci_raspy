import argparse
import multiprocessing as mp
import multiprocessing.shared_memory
import numpy as np
import yaml
import sys
import io
import os
import shutil
import time
import datetime
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
os.chdir('main') # todo: fix this import thing
import proc
import util
os.chdir('../')
period = 0.5 # seconds to sleep polling

parser = argparse.ArgumentParser()

parser.add_argument('model_name', help='Name of YAML file without extension describing the model')
parser.add_argument('--save', help='whether to save anything at all. WARNING: IF YOUR MODULES DEPEND ON data_folder, e.g. logger_disk, DO NOT SET THIS TO FALSE. Overrides --logfile. Default True.')
parser.add_argument('--logfile', help='whether to log print output to a logfile. Default True.')
models_dir = './models/' # todo?: maybe switch to dynamically allocated path or based on some install
modules_dir = './modules/'

args = parser.parse_args()
use_logfile = True if (args.logfile is None) or (args.logfile in ['True', 'true', '1']) else False
save_any = True if (args.save is None) or (args.save in ['True', 'true', '1']) else False
if not save_any:
    use_logfile = False
print('use_logfile', use_logfile)

# opening and loading the model declaration yaml file:
yaml_name = models_dir + args.model_name + '.yaml'
yaml_file = open(yaml_name, 'r')
yaml_data = yaml.load(yaml_file, Loader=Loader)
yaml_file.close()
signals = yaml_data['signals']
for name in signals:
    signals[name]['shape'] = eval(str(signals[name]['shape'])) # needed to convert tuple from str
modules = yaml_data['modules']

for name in modules:
    if 'name' not in modules[name]:
        modules[name]['name'] = name # Name of module Python file if name is not explicitly given.
            # Name of Python file can differ from module declaration (in yaml) if explicitly given.
    modules[name]['path'] = modules_dir + modules[name]['name']
        # i.e. the path of the module's loop file without .py
save_path = './data/'
if 'logger_disk' in modules and 'params' in modules['logger_disk'] and 'save_path' in modules['logger_disk']['params']:
    # Change the save_path if there's a logger_disk module which specifies otherwise.
    save_path = modules['logger_disk']['params']['save_path']
    
def main():
    # Format the data_folder with save_path based on time.
    # Possible change: change name based on model_name or argument
    now = datetime.datetime.now()
    now_str = now.strftime('exp_%Y-%m-%d_%H-%M-%S/')
    data_folder = save_path + now_str
    
    if save_any:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
    
    log_filename = data_folder + 'logfile.log'
    if use_logfile:
        log_file = open(log_filename, 'w')
        log_queue = mp.Queue(10000) # queue which holds the printed strings to be collected by main.py
        log_list = [] # list which holds the printed strings and is saved to disk at the end to prevent timing violations.
        # Class which captures print output. Writes to terminal and puts to log_queue.
        # Needed for the logfile to capture the print output
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
    else:
        log_queue = None
    
    for name in modules:
        if 'params' not in modules[name]:
            modules[name]['params'] = {}
        modules[name]['params']['data_folder'] = data_folder # overwrite data_folder for all modules...is this intended?
    if save_any:
        # Copy ./main/ and the module and modules used.
        # Why is this here? So you can go back and see the exact code that you ran.
        os.mkdir(data_folder + 'models/')
        shutil.copyfile(yaml_name, data_folder + 'models/' + args.model_name + '.yaml')
        os.mkdir(data_folder + 'main/')
        shutil.copyfile('./main/main.py', data_folder + '/main/main.py')
        shutil.copyfile('./main/proc.py', data_folder + '/main/proc.py')
        shutil.copyfile('./main/util.py', data_folder + '/main/util.py')
        os.mkdir(data_folder + 'modules/')
        for name in modules:
            module = modules[name]
            name = module['name'] # use target name for if name is an argument
            use_loop = module['loop'] if 'loop' in module else True
            if use_loop:
                shutil.copyfile(modules_dir + name + '.py', data_folder + 'modules/' + name + '.py')
            if 'constructor' in module and module['constructor']:
                shutil.copyfile(modules_dir + name + '_constructor.py', data_folder + \
                                'modules/' + name + '_constructor.py')
            if 'destructor' in module and module['destructor']:
                shutil.copyfile(modules_dir + name + '_destructor.py', data_folder + \
                                'modules/' + name + '_destructor.py')
    
    
    # Create all shared memory buffers for signals:
    shm_dict = {}
    for name, signal in signals.items():
        nbytes = int(np.prod(signal['shape'])*np.dtype(signal['dtype']).itemsize)
        try:
            shm_dict[name] = mp.shared_memory.SharedMemory(name=name, create=True, size=nbytes)
        except:
            tmp = mp.shared_memory.SharedMemory(name=name, create=False, size=nbytes)
            tmp.unlink()
            shm_dict[name] = mp.shared_memory.SharedMemory(name=name, create=True, size=nbytes)
    
    # Create communication pipes between processes in the directed graph.
    pipes = {name: {'in': [], 'out': [], 'trigger': None} for name in modules.keys()}
    triggers = {}
    for name, module in modules.items():
        if 'sync' in module:
            if module['sync'] is not None:
                for name2 in module['sync']:
                    conn1, conn2 = mp.Pipe()
                    pipes[name]['in'].append(conn1)
                    pipes[name2]['out'].append(conn2)
            else:
                module['trigger'] = True
        else:
            module['trigger'] = True # Give non-synced processes a trigger
    # Verify that directed graph is declared properly.
    if not util.verify_trigger(modules):
        raise ValueError('Every loop should have exactly one triggered process. '
                        'This implies that if there is a trigger in a branch, all other branches'
                        ' on the same level must also have a trigger.')
    # Create "trigger" pipes between main and triggered processes.
    for name, module in modules.items():
        if 'trigger' in module:
            if module['trigger']:
                conn1, conn2 = mp.Pipe()
                pipes[name]['trigger'] = conn1
                triggers[name] = conn2
        
    # Start the module processes.
    procs = {}
    for name, module in modules.items():
        procs[name] = mp.Process(target=proc.proc, args=(
            signals, module, pipes[name]['in'], pipes[name]['out'], pipes[name]['trigger'], log_queue), name=name)
        print('main {}\n'.format(name), end='')
        procs[name].start()
        time.sleep(0.01)
    
    # Send an empty string to trigger each triggered process
    for name, trigger in triggers.items():
        print('trigger {}\n'.format(name), end='')
        try:
            trigger.send('')
        except:
            pass # some exit i guess. to-do?
        
    try:
        while True:
            try:
                for name, trigger in triggers.items():
                    if trigger.poll(0.):
                        msg = trigger.recv()
                        if msg == 'quit_':
                            print('main: {} triggered quit\n'.format(name), end='')
                            raise Exception('someone quit')
                if log_queue is not None:
                    while not log_queue.empty():
                        line = log_queue.get()
                        #log_file.write(line)
                        log_list.append(line)
                time.sleep(period)
            except (Exception, KeyboardInterrupt) as e:
                print('!!! main Exception: {}\n'.format(str(e)), end='')
                for name, trigger in triggers.items():
                    try:
                        print('sending trigger quit for: {}\n'.format(name), end='')
                        trigger.send('quit_')
                    except:
                        pass
                break
    except:
        pass
    
    time.sleep(0.1)
    print('Begin termination of modules:\n', end='')
    # Termination of module processes
    for name in list(modules):
        try:
            procs[name].join(0.1)
            if procs[name].exitcode is None:
                procs[name].terminate()
                print('    Terminated: {}\n'.format(name), end='')
            else:
                print('    Joined: {}\n'.format(name), end='')
        except ValueError as e:
            print('    Not Joined: {}, {}\n'.format(name, str(e)), end='')
    for name in list(modules):
        try:
            procs[name].close()
        except ValueError as e:
            pass
    # Close shared memory buffers.
    for key, value in shm_dict.items():
        shm_dict[key].close()
        shm_dict[key].unlink()
    # Write print output to logfile (if applicable)
    if log_queue is not None:
        for line in log_list:
            log_file.write(line)
        while not log_queue.empty():
            log_file.write(log_queue.get())
    print('  Exiting main')
    return

if __name__ == '__main__':
    main()
