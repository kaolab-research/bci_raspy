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
import ast
import shlex
import pathlib
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

# Add parent directory to path.
# Useful for importing things.
raspy_dir = os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) )
sys.path.append( raspy_dir )

import proc2b as proc
import util2b as util

import re

def copy_path(src, dst):
    # See https://stackoverflow.com/questions/1994488/copy-file-or-directories-recursively-in-python/1994840#1994840
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
    except:
        shutil.copy2(src, dst)
    return

def main():
    os.chdir(raspy_dir)

    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])') # regex for remove ANSI escape sequences
    period = 0.5 # seconds to sleep polling

    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', help='Name of YAML file without extension describing the model')
    parser.add_argument('--save', help='whether to save anything at all. WARNING: IF YOUR MODULES DEPEND ON data_folder, e.g. logger_disk, DO NOT SET THIS TO FALSE. Overrides --logfile. Default True.')
    parser.add_argument('--logfile', help='whether to log print output to a logfile. Default True.')
    parser.add_argument('--data_folder', help='folder to save things in')
    parser.add_argument('--module_args', default='', help='params to pass on to modules as params->commandline_args, or as params->key. Format is --module_args "--global_var1 global_val1 /namespaceA --A_var1 A_val1 --A_var2 A_val2 /namespaceB" --B_var1 B_val1'
                        ' where namespace is the name of the module (the yaml name, which does not correspond to the file name if the name field is specified)')
    parser.add_argument('-overwrite_params', default=False, action='store_true', help='whether to overwrite params in addition to writing to commandline_args')
    #models_dir = './models/' # todo?: maybe switch to dynamically allocated path or based on some install
    #modules_dir = './modules/'
    models_dir = raspy_dir + '/models/'
    modules_dir = raspy_dir + '/modules/'

    args = parser.parse_args()
    use_logfile = False if (args.logfile in ['False', 'false', '0']) else True
    save_any = False if (args.save in ['False', 'false', '0']) else True
    if not save_any:
        use_logfile = False
    print('use_logfile', use_logfile)
    if args.module_args is None:
        raise ValueError('module_args cannot be None!')

    # opening and loading the model declaration yaml file:
    yaml_name = models_dir + args.model_name + '.yaml'
    yaml_file = open(yaml_name, 'r')
    yaml_data = yaml.load(yaml_file, Loader=Loader)
    yaml_file.close()
    signals = yaml_data['signals']
    for name in signals:
        signals[name]['shape'] = ast.literal_eval(str(signals[name]['shape'])) # needed to convert tuple from str
    modules = yaml_data['modules']
    
    '''
    # group_params are passed to module['params'] BEFORE commandline_args
    #   but AFTER loading the yaml
    #   WARNING: this will OVERWRITE module['params']
    # example
    group_params:
      # applies to all modules ONLY IF the name of the group is 'global'
      global:
        params:
          key: value
          key: value
      # otherwise, specify which modules should receive these params
      group1:
        modules:
          - module1
          - module2
        params:
          key: value
          key: value
      group2:
        ...
    '''
    group_params = yaml_data['group_params'] if 'group_params' in yaml_data else {}

    for name in modules:
        if 'name' not in modules[name]:
            modules[name]['name'] = name # Name of module Python file if name is not explicitly given.
                # Name of Python file can differ from module declaration (in yaml) if explicitly given.
        modules[name]['path'] = modules_dir + modules[name]['name']
            # i.e. the path of the module's loop file without .py
    # save_path is (probably) a directory named data that the run-specific data_folder resides within.
    #save_path = './data/'
    save_path = raspy_dir + '/data/'
    if 'logger_disk' in modules and 'params' in modules['logger_disk'] and 'save_path' in modules['logger_disk']['params']:
        # Change the save_path if there's a logger_disk module which specifies otherwise.
        save_path = modules['logger_disk']['params']['save_path']


    # Format the data_folder with save_path based on time.
    # Possible change: change name based on model_name or argument
    now = datetime.datetime.now()
    now_str = now.strftime('exp_%Y-%m-%d_%H-%M-%S/')
    if args.data_folder is None:
        data_folder = save_path + now_str
    else:
        data_folder = save_path + args.data_folder
        data_folder = data_folder.replace('{date}', now.strftime('%Y-%m-%d')) # Use {date} to automatically fill in the date

        # automatically increment counter. first value is 1.
        # Only works if pattern is in the tail of the data_folder.
        data_parent = os.path.dirname( os.path.abspath(data_folder) )
        df = os.path.basename(os.path.normpath(data_folder))
        if '{counter}' in data_folder:
            # max one instance per string
            pattern = df.replace('{counter}', '[0-9]+') + '$'
            matched_folders = [folder for folder in os.listdir(data_parent) if re.match(pattern, folder) is not None]
            counter_vals = []
            for folder in matched_folders:
                l = df.index('{counter}')
                counter_start = re.match(df[0:l], folder).span()[1]
                counter_length = re.match('[0-9]+', folder[counter_start:None]).span()[1]
                counter_val = int(folder[counter_start:None][0:counter_length])
                counter_vals.append(counter_val)
            if len(counter_vals) == 0:
                data_folder = data_folder.replace('{counter}', '1')
            else:
                data_folder = data_folder.replace('{counter}', str(max(counter_vals) + 1))
    if data_folder[-1] != '/':
        data_folder = data_folder + '/'
    
    if save_any:
        if not os.path.exists(save_path):
            pathlib.Path(save_path).mkdir(parents=True)
        if not os.path.exists(data_folder):
            pathlib.Path(data_folder).mkdir(parents=True)
    
    log_filename = data_folder + 'logfile.log'
    if use_logfile:
        log_file = open(log_filename, 'w')
        log_file.write(str(sys.argv) + '\n') # write the command line arguments to the logfile
        
        log_queue = mp.Queue(10000) # queue which holds the printed strings to be collected by main.py
        log_list = [] # list which holds the printed strings and is saved to disk at the end to prevent timing violations.
        # Class which captures print output. Writes to terminal and puts to log_queue.
        # Needed for the logfile to capture the print output
        class Logger(object):
            def __init__(self):
                self.terminal = sys.stdout # Get and save (a reference to) the original stdout/print stream.

            def write(self, message):
                if 'warning' == message[0:7].lower():
                    message = util.color_str(message, (255, 255, 0))
                self.terminal.write(message) # Actually print the stdout message.
                message = ansi_escape.sub('', message) # remove ANSI color codes
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
        if not isinstance(modules[name]['params'], dict): # deal with when the param list is empty
            modules[name]['params'] = {}
        modules[name]['params']['data_folder'] = data_folder # overwrite data_folder for all modules...is this intended?
        modules[name]['params']['_raspy_dir'] = raspy_dir # directory containing the main subdirectory.
        modules[name]['params']['commandline_args'] = {}
    for group_name, group in group_params.items():
        if group_name == 'global':
            modules = modules.keys()
        for name in modules:
            for param, value in group['params'].items():
                modules[name]['params'][param] = value
                pass

    # Parse and add module_args into respective module['params']['commandline_args']
    namespace_module_args = '/global'
    module_args_state = 'no_name'
    for word in shlex.split(args.module_args):
        if word[0] == '/':
            if len(word) == 1:
                raise ValueError(f'module name/namespace is empty')
            namespace_module_args = word[1:]
        elif word[0:2] == '--' and module_args_state == 'no_name':
            module_args_state = 'has_name'
            if len(word) == 2:
                raise ValueError(f'argument name is empty')
            module_arg_name = word[2:]
        elif module_args_state == 'has_name':
            module_arg_value = word
            if namespace_module_args == '/global':
                for name in modules:
                    modules[name]['params']['commandline_args'][module_arg_name] = module_arg_value
                    if args.overwrite_params:
                        modules[name]['params'][module_arg_name] = module_arg_value
            else:
                if namespace_module_args not in modules:
                    raise ValueError(f'namespace {namespace_module_args} is not a valid module')
                modules[namespace_module_args]['params']['commandline_args'][module_arg_name] = module_arg_value
                if args.overwrite_params:
                    modules[namespace_module_args]['params'][module_arg_name] = module_arg_value
            module_args_state = 'no_name'
        else:
            raise ValueError('--module_args is improperly formatted.')

    if save_any:
        # Copy ./main/ and the module and modules used.
        # Why is this here? So you can go back and see the exact code that you ran.
        dest_yaml_path = data_folder + 'models/' + args.model_name + '.yaml'
        os.makedirs(os.path.dirname( os.path.abspath(dest_yaml_path) ), exist_ok=True)
        shutil.copyfile(yaml_name, dest_yaml_path)
        os.mkdir(data_folder + 'main/')
        shutil.copyfile(raspy_dir + '/main/main2.py', data_folder + '/main/main2.py')
        shutil.copyfile(raspy_dir + '/main/proc2.py', data_folder + '/main/proc2.py')
        shutil.copyfile(raspy_dir + '/main/util2.py', data_folder + '/main/util2.py')
        shutil.copyfile(raspy_dir + '/__init__.py', data_folder + '/__init__.py')
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
    
        paths_to_copy = yaml_data['paths_to_copy'] if 'paths_to_copy' in yaml_data else [] # a list of relative paths (files or dirs) to copy
        for path_name in paths_to_copy:
            if path_name[0] == '.' or '..' in path_name:
                try:
                    copy_path(path_name, data_folder + path_name)
                except:
                    print(util.color_str(f'Operation to copy {path_name} failed. Skipping.', (255, 0, 0)))
            else:
                print(util.color_str(f'Path is not relative or path references a parent directory! Skipping {path_name}', (255, 0, 0)))
            pass
    
    # Create all shared memory buffers for signals within a manager:
    #   This helps make sure all shared memory gets shut down.
    pid = os.getpid()
    shared_memory_manager = util.RaspySharedMemoryManager(signals)
    print(f'pid: {pid}\n', end='')
    
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
    
    def flush_log(lg_queue, lg_list):
        if lg_queue is not None:
            while not lg_queue.empty():
                line = lg_queue.get()
                lg_list.append(line)
        return

    try:
        # Start the module processes.
        procs = {}
        for name, module in modules.items():
            procs[name] = mp.Process(target=proc.proc, args=(
                signals, module, pipes[name]['in'], pipes[name]['out'], pipes[name]['trigger'], log_queue, shared_memory_manager.shm_names), name=name)
            print('main {}\n'.format(name), end='')
            procs[name].start()
            time.sleep(0.01)
        
        # Send an empty string to trigger each triggered process
        for name, trigger in triggers.items():
            print('trigger {}\n'.format(name), end='')
            try:
                trigger.send('')
            except:
                print('failed to send trigger for {}\n.'.format(name), end='')
                pass # some exit i guess. to-do?


        # catch messages from trigger modules.
        # quit if receiving 'quit_'
        while True:
            try:
                for name, trigger in triggers.items():
                    if trigger.poll(0.):
                        msg = trigger.recv()
                        if msg == 'quit_':
                            print('main: {} triggered quit\n'.format(name), end='')
                            raise Exception('someone quit')
                flush_log(log_queue, log_list)
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
    finally:
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
        
        print('  Exiting main')
        flush_log(log_queue, log_list)
        # Write print output to logfile (if applicable)
        if log_queue is not None:
            for line in log_list:
                log_file.write(line)
        del shared_memory_manager
    return

if __name__ == '__main__':
    main()
