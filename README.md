# Experimental code for "Brain-machine interface control with artificial intelligence copilots"
Traditional brain-machine interface (BMI) systems decode movement solely from neural data, without taking advantage of goal information, including the locations of potential target locations.
The CNN-KF (convolutional neural network-Kalman filter) and artificial intelligence (AI) copilot take advantage of task structure by updating decoder parameters in closed loop (CNN-KF) and changing the distribution of actions based on observations of the environment that the user is engaging with (copilot).
We provide code for model training, as well as some experimental code, along with some demonstration data.
Plotting and analysis code can be found in the `bci_plot` repository (may be subject to a name change).

## Requirements:
Python 3.8 (Used 3.8.18)
Tested with Ubuntu 20.04/22.04
A CUDA-enabled GPU is recommended for neural network training (with PyTorch installed with GPU enabled).
See `requirements.txt` for installed packages. Installation should take less than 1 hour, depending on download speeds. PyTorch may require device-specific installation according to https://pytorch.org/get-started/locally/.
`sb3-contrib==1.6.2` and `stable_baselines==1.6.2` require `setuptools==65.5.0` and `wheel==0.38.4` due to a dependency on `gym==0.21`, which may require separate installation from other packages.
To install all packages in `requirements.txt`, run
```pip install -r requirements.txt```
in your desired virtual environment.

Real-time data collection requires EEG streaming software @ 1000Hz, 64+3 channels.
We used ANT Neuro's SDK version 1.3.19 to stream data from the eego rt amplifier using custom C++ code (`stream/stream_data.cc`). Running this code requires proprietary library files.
We additionally provide a function `stream_fake.py` to emulate EEG streaming but instead with random noise, which can be invoked with `python stream/stream_fake.py --ip 127.0.0.1 --port 7779 --fs 1000.0 --dt 0.001 --std 0.00001`, for example.
Most code is configured to run with an NVIDIA GPU using PyTorch's CUDA interface.

Gaze data is received at 60Hz from the Tobii Pro Nano from a separate Windows machine.
Gaze positions were normalized to the monitor used during experiments, then rescaled to match its resolution.
The names of all gaze data points collected are enumerated in `modules/recv_gaze_buffer_constructor.py`, though we only use `left_gaze_point_on_display_area` and `left_gaze_point_on_display_area` for analysis.

## CNN-KF initial training
To demo training of the convolutional neural network (CNN) and initial training of the Kalman filter (KF), the below command should be run (expected execution time 10-20 minutes with CPU).
```
cd Offline_EEGNet
python pipeline_kf_func.py config.yaml
```
Users can also run the training using a CUDA GPU using `config_gpu.yaml` instead.
This demo uses a truncated version of a 1D decorrelated task `data/raspy/demo_decorr` (2.8 minutes rather than 20 minutes).
As a result, data labeling is unbalanced and poor results should be expected (reference results can be found in `data/raspy/trained_models/reference_demo_decorr_cpu` and `data/raspy/trained_models/reference_demo_decorr_gpu`)

## KF adaptation
Code used for adaptation of the Kalman Filter can be found in `modules/kf_util.py`, `modules/kf_clda_constructor.py`, and `modules/kf_clda.py`.
Due to the non-standard nature of so-called modules in the real-time framework RASPy, these modules should not be run directly.
A description of RASPy is provided later in this README.

We provide some code to demonstrate (but not recreate) the data of `data/raspy/demo_centerout`, using previously recorded EEG data.
This requires pygame to create a new window.
From this directory, users can run
```
python main/main2b.py replay_demo
```
whose output should somewhat (but not exactly) resemble demo_centerout_replayed.
This demonstrates the very first center-out task done by participants; The large target size is only used for one 10-minute session.
`data/raspy/demo_centerout_replayed/xy_diff.png` shows the difference in `x` and `y` cursor positions between `demo_centerout` and `demo_centerout_replayed` due to differences in code execution at runtime.
This is mainly due to delta time processing of real-world velocities, rather than using tick-wise real-world velocities. 
Of note, the cursor trajectories diverge after the regression Kalman Filter parameters (named M1 and M2) are updated.

## Copilot training
To demo training of the LSTM copilot, run
```
python -m SJtools.copilot.train -model=RecurrentPPO -batch_size=512 -action chargeTargets -action_param temperature 1 -obs targetEnd -holdtime=2.0 -stillCS=0.0 -lr_scheduler=constant -softmax_type=normal_target -velReplaceSoftmax -no_wandb -reward_type=baseLinDist.yaml -center_out_back -extra_targets_yaml=dir-8.yaml -timesteps=2048 -n_steps=512 -renderTrain
```
from this directory. The `-renderTrain` flag enables a display of the environment during trajectory collection.
It is normal for this display to stop updating for several seconds while parameters are updated.
The value of `-timesteps` is intentionally low so that the demo can be run in a short amount of time (1-4 minutes).
Each training generates a directory under `SJtools/copilot/runs`.

A visualized test of the copilot can be shown using
```
python -m SJtools.copilot.test models/keep/charge/T8B_LSTM2_truedecay/best_model -center_out_back -softmax_type=normal_target
```
More examples and features can be found at `SJtools/README`

## Experimental code
Experimental RASPy models in `models/exp` (see #bci_raspy) were generated by `experiments/generate_sh.py` based on the templates in `models/templates`.
The experiment `.sh` files started the RASPy models along with EEG streaming processes.
After beginning each experiment `.sh` file in `experiments/scripts`, on an external windows machine, we ran Tobii software to send gaze data (processed by the `recv_gaze_buffer` RASPy module).
These scripts are not intended to be run as a demonstration since the streaming requires installation of external code, and many file paths are absolute.
An example of the text experiment can be seen by running
```
python main/main2b.py exp/SJ-text-gaze_demo
```
(press escape to quit).


# bci_raspy
Repository for RASPy implementations of BCI tasks.
The RASPy (Realtime Asynchronous or Synchronous Python) framework is a pure Python framework for running modular real-time experimental code.
It is based off of the LiCoRICE project (https://github.com/bil/licorice), and has fewer dependencies and several feature changes.
Fundamentally, models consist of modules and signals; modules define the code that is run while signals represent data transferred between modules.
The looping functionality of and synchronization between modules enable real-time experimental data collection.

## Running a model
RASPy can be run from any directory, but uses the raspy directory as its working directory (and so do all modules).
Example running from bci_raspy directory:
```
python ./main/main.py $model_name
```
Note: replace the dollar sign along with model_name. The model name is the relative path of a .yaml file within the `models` directory, without the file extension.

## Writing models
Standard modules: 
* logger_disk (Used to be able to be run separately, lmk if this is desired)
* timer
* UpdateEEG
* filterEEG
* task_module
* decoder
* logger
* task_pygame_module

Use the sync entry to describe the timing dependency directed graph. Use trigger=True to specify which modules should be run first when the model is started. Use the params entry, which will be directly accessible within the module as a variable `params`, to provide module-specific parameters. Be sure to update in and out signal lists (same function, use just one if desired).

## Writing modules
If you create a variable `quit_` and it evaluates to True within any module, this will signal RASPy to quit the entire model. This is preferred over Ctrl-C.
Sadly, no threading/sub/multiprocessing allowed unless they start and end completely in the constructor, in the destructor, or in a single loop.

IMPORTANT: raspy_dir/main/main.py changes the working directory to raspy_dir. This allows `from main import utils`, `from decoders import EEGNet`, `from modules import some_util_module`, etc, but will affect relative paths within each model.

e.g.
```
  timer:
    #name: timer # this references which .py files are used. If not specified, the name of the module itself (prev line) is used instead.
    constructor: True
    destructor: True
    sync:
      - logger
    trigger: True
    params:
      timer_type: sleep # pygame or sleep
      dt: 1000 # in microseconds.
```

By default, the name of the module itself is used to reference the `modulename_constructor.py`, `modulename.py` (loop), and `modulename_destructor.py` parts. If you specify a `name` field, then the files corresponding to `name` will be used instead.

`loop` defaults to True whereas `constructor` and `destructor` default to False!!!

## Standard module descriptions and tips
### logger_disk
Goes with logger. Saves data transmitted from logger into respective streams within the params['save_path'] directory. Creates a new folder based on the current time. 

e.g.
```
  # WARNING: make sure you have an exit condition when using loop: False
  # Note: it is possible for KeyboardInterrupt to be sufficient.
  logger_disk:
    constructor: True
    destructor: False
    loop: False
    params:
      save_path: ./data/ # optional, stored in save_path of main script. Relative is relative to raspy_dir. This affects the data_folder of all modules. default raspy_dir + '/data/'
      connections:
        local: # connection_name. Usually only have 1. Not sure why this level is here.
          task: # stream_name
            IP: '127.0.0.1'
            PORT: 7701
          eeg:
            IP: '127.0.0.1'
            PORT: 7702
```

### timer
Maintains real-world clock synchronicity of models. Resolution is ~0.1ms standard deviation. params['timer_type']: sleep (recommended for UNIX, though hybrid may be more precise), pygame, busy, or hybrid (i.e. sleep then busy). hybrid can overshoot if time_sleep_buffer is too small (set to 1ms). params['dt']: desired period in microseconds.

### UpdateEEG
Receives EEG samples from separate stream (see bci_ant_streaming, 66 double-precision values per sample) and stores them in eegbuffersignal (bipartite buffer). Stores the next index in eegbufferindex (always > 0.5\*length of buffer). Please ensure that eegbuffersignal is long enough so that there is no index is written twice in a single cycle.

### filterEEG
Filters data stored in eegbufferindex and outputs it to databuffer. Currently uses a linear filter as specified in filterEEG_constructor.py in SOS format.

### task_module
Manipulate task variables here. Also set taskbufferindex to eegbufferindex at each cycle (reason: synchronicity, not really an amazing reason).

### decoder
Imports the decoder_name decoder from decoder_folder. 
to-do: figure out how to handle batch/no batch dimension

### logger
Sends streams of data to different connections. Specify an IP and PORT for each stream for each connection. params['log'] specifies the stream_name, and either signals or index+buffers (can't have both signals and buffers) to send to each stream. Each stream also sends the tick/step of all other streams (for synchronizing streams afterward).

This should agree with the `logger_disk` module if you're saving within raspy. Different behavior: sequential pickling of variables (flag: True when saving, records: pickled bytes, lengths: length of the pickled data). Alternatively, just pickle it yourself? Not sure.

Set `pause` (name of signal/variable) and `pause_condition` (any or all) if you want to pause the logger sometimes. I couldn't find a clean way to pause the entire state without risking overflowing buffers. Note that the logger's actual pause state lags the pause variable by one tick.

e.g.
```
  logger:
    constructor: True
    destructor: True
    sync:
      - decoder
    in:
      - eegbuffersignal
      - databuffer
      - eegbufferindex
      - state_task
      - v_decoded
      - decoder_output
    params:
      #pause: name_of_pause_signal
      #pause_condition: any # any: pause if any of name_of_pause_signal evals True. all: '' if all '' evals False
      log:
        task:
          signals:
            - state_task
            - decoder_output
        eeg:
          index: eegbufferindex
          buffers:
            - eegbuffersignal
            - databuffer
        #pickled:
        #  flag: pickle_flag # bool
        #  records:
        #    - record1_name
        #    - record2_name
        #  lengths:
        #    - record1_length
        #    - record2_length
      connections:
        local:
          task:
            IP: '127.0.0.1'
            PORT: 7701
          eeg:
            IP: '127.0.0.1'
            PORT: 7702
          #pickled:
          #  IP: '127.0.0.1'
          #  PORT: 55555
```

### task_pygame_module
Update the task feedback based on task variables. Expect this to take several milliseconds (>2.3ms) each update.

### utility importables
These are not raspy modules but are importable as `from modules import name` or simply `import name`

#### buffer_util
Contains DoubleCircularBuffer, which is a medium-level interface for accessing double circular buffers.
Usage:
```
from buffer_util import DoubleCircularBuffer

my_databuffer = DoubleCircularBuffer(databuffer, eegbufferindex)
eeg_window_length = 1000 # in samples
# getting a window
eeg_window = my_databuffer.pull(eeg_window_length)[:, channels_to_keep]

# getting new samples
new_eeg = my_databuffer.read(copy_data=False) # shape (num_new_samples, 66)

# writing
my_databuffer.write(gen_eeg) # where gen_eeg has shape (num_new_samples, 66)
```

#### data_filter_util
Contains class `DataFilter`, which can be used to filter data either offline (filter_data) or online (filter_online)

## Group and global params
Specify these on the same level as signals and as modules. global affects all modules. other groups will only affect the specified modules.
IMPORTANT: This overwrites the params that are specified on the individual module level.

```
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
    ..........
```

## loading the logged data
use main-> `data_util` -> `load_data` on the `.bin` file. By default, returns a dict. Use `copy_arr=True` to copy the arrays before returning (this is useful to free up memory when downsampling)

## Misc

### debugging
if you are using ubuntu 22.10 + miniconda, you might run into problem because of lib std path problem.

here is fix that can be done foud from https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris:


$ cd /home/$USER/miniconda2/lib
$ mkdir backup  # Create a new folder to keep the original libstdc++
$ mv libstd* backup  # Put all libstdc++ files into the folder, including soft links
$ cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6  ./ # Copy the c++ dynamic link library of the system here
$ ln -s libstdc++.so.6 libstdc++.so
$ ln -s libstdc++.so.6 libstdc++.so.6.0.19


