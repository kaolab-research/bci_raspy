from utils import data_file_to_dict, read_config
from preprocessor import DataPreprocessor
import numpy as np
from scipy.signal import resample

class PreprocessLikeClosedLoop:
    '''This class handles takes one or more data files and transforms them into 
    a shape where a model can evaluate them like raspy does, i.e., tick by tick.

    Uses a dictionary of configuration settings which is generally intended to 
    be created from a yaml file.

    Example
    -------
    preprocessor = PreprocessLikeClosedLoop(config_dict)
    eeg_trials, eeg_trial_labels = preprocessor.preprocess(eeg_data, task_data)
    '''

    def __init__(self, config):
        '''Initializes DataPreprocessor with given arguments from the config file.

        Parameters
        ----------
        config: dict
            Configurations from the yaml file.

        self.first_run is used to flag that the preprocess function has not yet 
        been run for online experiments.
        '''

        self.eeg_cap_type = config['eeg_cap_type']
        self.ch_to_drop = config['ch_to_drop']
        self.labels_to_keep = config['labeling']['labels_to_keep']
        self.relabel_pairs = config['labeling']['relabel_pairs']
        
        self.sf = config['sampling_frequency']
        self.online_status = config['online_status']
        self.normalizer_type = config['normalizer_type']
        self.first_run = True



def preprocess_like_closed_loop(data_name, 
                                eeg_cap_type='gel64',
                                ch_to_drop=['TRGR', 'COUNT', 'F7', 'F5', 'F3', 'F1', 
                                     'Fz', 'F2', 'F4', 'F6', 'F8', 'Fp1', 
                                     'Fpz', 'Fp2', 'AF7', 'AF3', 'AF4', 
                                     'AF8', 'EOG'],
                                labels_to_keep=['all'],
                                relabel_pairs=None,
                                train_on_server=True,
                                initial_ticks=100,
                                latency=1000,
                                z_scorer='Welfords',
                                min_std=0,
                                detect_artifacts=True,
                                reject_std=5.5):
    '''This function preprocesses data in the same way it happens online in a 
    closed loop experiment. In other words, it iterates over the data tick by 
    tick an processes it using only data available at that point in time.

    data_name should be a folder name within which eeg and task data is stored from raspy.
    
    Drops all front channels by default to avoid eye movement artifacts, as well as EOG.

    If train_on_server = True, data must be stored in /data/raspy for function to work.

    If train_on_server = False, data must be stored in Offline_EEGNet/data for function to work.
    
    eeg_cap_type can be gel64, dry64, or saline64.

    labels_to_keep should be a list of ints corresponding to the labels we want 
    to use as encoded in the raw data from bci_raspy. If using the default ['all'],
    model will be trained to predict all labels in the data.

    relabel_pairs is a list of tuples used to simulate when tasks are remapped 
    during closed loop experiments. For instance, in a closed loop experiment if 
    the mental math task was used to move the cursor right in bci_raspy settings 
    the label for that task in the closed loop data would be a 1 (the usual 
    label for left), while in the training data it would be a 4 (the usual label 
    for math). To correct for this, pass [(1, 4)] into relabel pairs, so that 
    task labels of 1 in the test dataset will be reassigned to be 4 (to match 
    the corresponding training data).
    
    latency is the number of milliseconds of backward-looking data to use at each tick.

    z_scorer is the normalizer to use at each tick. Default is 'Welfords' which 
    calculates the true mean and standard deviation at each tick. Alternative is 
    'Running_Mean' which more heavily weights more recent samples.

    min_std is the minimum std to use when normalizing at each tick.

    reject_std is the number of standard deviations from the running mean and std at that point to use 
    when determining whether a window of data includes an artifact and should be excluded.
    
    initial_ticks is the number of ticks that must accumulate before artifact 
    detection begins and bad windows begin to be skipped.'''

    #Folder to look for data to use and where to save completed preprocessed data
    data_folder, dest_folder = get_data_dest(train_on_server)
    
    # load in data
    eeg_data = data_file_to_dict(data_folder + data_name + "/eeg.bin")
    task_data = data_file_to_dict(data_folder + data_name + "/task.bin")
    #Some data has shape XXXX, 1 instead of flat state_task - flatten if so
    if task_data['state_task'].ndim == 2:
        task_data['state_task'] = task_data['state_task'].flatten()
    #If we are relabeling any labels, swap them to desired values
    if relabel_pairs:
        for pair in relabel_pairs:
            label_to_remove = pair[0]
            label_to_place = pair[1]
            task_data['state_task'] = np.array([label_to_place if 
                                                label == label_to_remove else 
                                                label for label in 
                                                task_data['state_task']])


    #Get channel labels and indexes to drop
    labels, coords, ch_index_to_drop = get_electrode_position(eeg_cap_type, ch_to_drop)
    
    #Delete the channels we are dropping
    eeg_data['databuffer'] = np.delete(eeg_data['databuffer'], 
                                       ch_index_to_drop, 
                                       1)
    
    #Apply laplacian filter
    eeg_data['databuffer'] = next_next_neighbors_filter(eeg_data['databuffer'], 
                                                        eeg_cap_type=eeg_cap_type, 
                                                        ch_to_drop=ch_to_drop)

    # Create flag variable to show normalizer object not yet instantiated
    normalizer_started = False
    #Create counter to track how many artifact windows were detected
    artifact_counter = 0

    #Create list of bad labels we don't want to consider
    all_labels = np.unique(task_data['state_task'])
    #Add intertrial periods to list of bad labels
    bad_labels = [-1]
    #Add other labels we want to exclude to bad labels
    if labels_to_keep != ['all']:
        also_bad = [label for label in all_labels.list() if label not in labels_to_keep]
        bad_labels = bad_labels + also_bad

    #Create empty lists to hold the data we will return from function
    eeg_trials = []  # will hold each trial of eeg data
    eeg_trial_labels = [] # will hold the label for each trial

    #Iterate across the ticks of the test dataset and process them
    print('iterating across all ticks to preprocess like closed loop')
    for tick in range(task_data['eeg_step'].shape[0]):
        end_ix = task_data['eeg_step'][tick]

        #If no data in databuffer, all 0s to match online behavior
        if end_ix <= 0:
            data = np.zeros([latency, eeg_data['databuffer'].shape[1]])
        
        #If some data in databuffer, supplement w/0s to match online behavior
        elif end_ix < latency:
            #Get all the data we do have from the eeg data buffer
            data_end = eeg_data['databuffer'][:end_ix, :]
            #Fill rest of latency with zeros
            data_start = np.zeros([latency-end_ix, eeg_data['databuffer'].shape[1]])
            data = np.concatenate((data_start, data_end), axis=0)                
        
        #If we have enough data in the databuffer, use it
        else:
            #Extract the latency period - that decoder would see closed loop
            start_ix = end_ix - latency
            data = eeg_data['databuffer'][start_ix:end_ix, :]
        
        #Set flag variable for whether artifact detected in current tick
        artifact=False
        
        #Create welfords running mean if not yet running
        if not normalizer_started:
            if z_scorer == 'Welfords':
                normalizer = Welfords(data)
                normalizer_started = True
            if z_scorer == 'Running_Mean':
                normalizer = Running_Mean(data)
                normalizer_started = True

        #Adjust counter; check if min ticks passed to start artifact detection
        #Detect artifacts function also adds data to normalizer if it is good
        initial_ticks -= 1
        if initial_ticks <= 0 and detect_artifacts:
            artifact = detect_artifacts_closed(data, 
                                               normalizer, 
                                               reject_std=reject_std)

        #If not enough ticks or not detecting artifacts, add data to normalizer
        if initial_ticks > 0 or not detect_artifacts:
            normalizer.include(data)
                                
        #If artifact detected, count artifact and go to next tick
        if artifact:
            artifact_counter += 1
            continue

        #If not an artifact, add data to running mean and record for decoding
        if not artifact:
            #Check if in intertrial period or label we don't want, if so skip
            if task_data['state_task'][tick] in bad_labels:
                continue
            #Normalize data
            data = ((data-normalizer.mean) / 
                    np.maximum(normalizer.std, np.full(normalizer.std.shape, min_std)))
            #Downsample each trial from 1000Hz to 100Hz
            data = resample(data, int(latency/10), axis=0)
            
            #Save data to our running lists
            eeg_trials.append(data)
            eeg_trial_labels.append(task_data['state_task'][tick])
    
    #Calculate share of ticks rejected as artifacts
    total_ticks = len([label for label in task_data['state_task'] if label in bad_labels])
    artifact_percent = artifact_counter / total_ticks

    #Convert data and labels to arrays
    eeg_trials = np.array(eeg_trials)
    eeg_trial_labels = np.array(eeg_trial_labels)

    #Return share of ticks rejected as artifacts, eeg_data, and labels
    print(f'Share of trials that were rejected as artifacts = {artifact_percent}')
    return artifact_percent, eeg_trials, eeg_trial_labels
    


def detect_artifacts_closed(eeg_data,
                            normalizer,
                            reject_std=5.5):
    '''This function checks eeg data for artifacts. In each channel it looks for outliers based on the backward-looking 
    standard deviation and mean.

    eeg data should be the most recent window of eeg data that is about to be considerd by the decoder. Should 
    be shape (samples, channels).

    normalizer should be an instantiated Welfords or Running_Mean class object to calc the ongoing mean and std dev.
    
    reject_std is the number of standard deviations to allow each channel to vary by. If any data points in a window exceed 
    this threshold, the window will be marked as containing an artifact.
    
    It returns True if an artifact is detected, otherwise False.
    
    '''
    
    #Set flag for whether this window is bad data
    bad_window = False
    #Get mean and std dev for previous samples up to this point
    mu = normalizer.mean
    std = normalizer.std
    #Iterate across all the data points in the data and check if any channels exceed rejection threshold
    for i in range(eeg_data.shape[0]):
        #If already found bad window, end this loop
        if bad_window:
            break
        #Check each channel at this timepoint
        deviations = abs(eeg_data[i,] - mu) / std
        if any(deviations > reject_std):
            bad_window = True
    #If window is not bad, add this window to the running mean and std
    if not bad_window:
        normalizer.include(eeg_data)

    return bad_window