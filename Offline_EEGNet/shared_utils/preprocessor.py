import numpy as np
from scipy.signal import butter
from scipy import signal
import sys
try:
    from . import utils
except:
    import utils

class DataPreprocessor:
    '''This preprocessor handles the general preprocessing work, including dropping channels, several filtering, and normalization.

    Example
    -------
    preprocessor = DataPreprocessor(config_dict)
    eeg_data['databuffer'] = preprocessor.preprocess(eeg_data['databuffer'])
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
        self.apply_bandpass = config['bandpass_filter']['apply']
        self.lowcut = config['bandpass_filter']['lowcut']
        self.highcut = config['bandpass_filter']['highcut']
        self.order = config['bandpass_filter']['order']
        self.sf = config['sampling_frequency']
        self.online_status = config['online_status']
        self.normalizer_type = config['normalizer_type']
        self.first_run = True
        self.zero_center = bool(config.get('zero_center', True)) # True if subtracting the mean of data. Only applies to offline and Welfords, NOT running_mean.

    def get_electrode_position(self):
        '''Get electrode names and grid coordinates for different cap types.

        Returns
        -------
        ch_names: list of string
        coords: list of integer pairs (list)
            Position indices of each electrode in the grid.
        '''

        if self.eeg_cap_type == 'gel64':
            ch_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
                        'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6',
                        'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2', 'EOG',
                        'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6',
                        'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CP4',
                        'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6',
                        'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Oz',
                        'TRGR', 'COUNT']
            coords = [[0, 4], [0, 5], [0, 6], [2, 1], [2, 3], [2, 5], [2, 7], [2, 9],
                      [3, 2], [3, 4], [3, 6], [3, 8], [4, 0], [4, 1], [4, 3], [4, 5], [4, 7], [4, 9], [4, 10], 
                      [5, 2], [5, 4], [5, 6], [5, 8], [6, 1], [6, 3], [6, 5], [6, 7], [6, 9],
                      [7, 5], [8, 4], [8, 6], [0, 0], [1, 1], [1, 3], [1, 7], [1, 9],
                      [2, 2], [2, 4], [2, 6], [2, 8], [3, 3], [3, 5], [3, 7],
                      [4, 2], [4, 4], [4, 6], [4, 8], [5, 3], [5, 7],
                      [6, 2], [6, 4], [6, 6], [6, 8],
                      [7, 2], [7, 3], [7, 7], [7, 8],
                      [3, 1], [3, 9], [5, 1], [5, 9], [7, 1], [7, 9], [8, 5],
                      [4, 8], [8, 5]]
        elif self.eeg_cap_type == 'dry64':
            ch_names = ['0Z', '1Z', '2Z', '3Z', '4Z',
                        '1L', '1R', '1LB', '1RB', 
                        '2L', '2R', '3L', '3R', '4L', '4R', 
                        '1LC', '1RC', '2LB', '2RB', '1LA', '1RA', '1LD', '1RD', '2LC', '2RC', 
                        '3LB', '3RB', '3LC', '3RC', '2LD', '2RD', '3RD',
                        '3LD', '9Z', '8Z', '7Z', '6Z', '5Z', 
                        '10L', '10R', '9L', '9R', '8L', '8R', '7L', '7R', '6L', '6R', '5L', '5R', 
                        '4LD', '4RD', '5LC', '5RC', '5LB', '5RB', 
                        '3LA', '3RA', '2LA', '2RA', '4LC', '4RC', '4LB', '4RB',
                        'TRGR', 'COUNT']
            coords = [[0, 5], [2, 5], [4, 5], [6, 5], [8, 5],
                      [1, 4], [1, 6], [5, 2], [5, 8],
                      [3, 4], [3, 6], [5, 4], [5, 6], [7, 4], [7, 6],
                      [5, 1], [5, 9], [7, 2], [7, 8], [8, 3], [8, 7], [6, 0], [6, 10], [7, 1], [7, 9],
                      [9, 2], [9, 8], [9, 1], [9, 9], [8, 0], [8, 10], [10, 10],
                      [10, 0], [18, 5], [16, 5], [14, 5], [12, 5], [10, 5],
                      [19, 4], [19, 6], [17, 4], [17, 6], [15, 4], [15, 6], [13, 4], [13, 6], [11, 4], [11, 6], [9, 4], [9, 6],
                      [12, 0], [12, 10], [13, 1], [13, 9], [13, 2], [13, 8],
                      [12, 3], [12, 7], [10, 3], [10, 7], [11, 1], [11, 9], [11, 2], [11, 8],
                      [19, 10], [19, 0]]
        elif self.eeg_cap_type == 'saline64':
            ch_names = ['1Z', '2Z', '3Z', '4Z', '6Z', '7Z', '8Z', '9Z',
                        '1L', '2L', '3L', '4L', '5L', '6L', '7L', '8L', '9L', '10L', '11L',
                        '1R', '2R', '3R', '4R', '5R', '6R', '7R', '8R', '9R', '10R', '11R',
                        '1LA', '2LA', '3LA', '1LB', '2LB', '3LB', '4LB', '5LB', '1LC', '2LC', '3LC', '4LC', '5LC',
                        '1LD', '2LD', '3LD', '4LD',
                        '1RA', '2RA', '3RA', '1RB', '2RB', '3RB', '4RB', '5RB', '1RC', '2RC', '3RC', '4RC', '5RC',
                        '1RD', '2RD', '3RD', '4RD',
                        'TRGR', 'COUNT']
            coords = [[2, 5], [4, 5], [6, 5], [8, 5], [12, 5], [14, 5], [16, 5], [18, 5],
                      [1, 4], [3, 4], [5, 4], [7, 4], [9, 4], [11, 4], [13, 4], [15, 4], [17, 4], [19, 4], [21, 4],
                      [1, 6], [3, 6], [5, 6], [7, 6], [9, 6], [11, 6], [13, 6], [15, 6], [17, 6], [19, 6], [21, 6],
                      [8, 3], [10, 3], [12, 3], [5, 2], [7, 2], [9, 2], [11, 2], [13, 2], [5, 1], [7, 1], [9, 1], [11, 1], [13, 1],
                      [6, 0], [8, 0], [10, 0], [12, 0],
                      [8, 7], [10, 7], [12, 7], [5, 8], [7, 8], [9, 8], [11, 8], [13, 8], [5, 9], [7, 9], [9, 9], [11, 9], [13, 9],
                      [6, 10], [8, 10], [10, 10], [12, 10],
                      [10,  5], [0, 5]]
        else:
            raise ValueError('eeg_cap_type must be one of "gel64", "dry64", "saline64".')
        return ch_names, coords

    def bandpass_channels(self, data):
        '''Apply butter bandpass filter. Both high pass band and low pass band can be assigned.

        Parameters
        ----------
        data: 2-d array with shape (n_samples, n_electrodes)

        Returns
        -------
        data: 2-d array with shape (n_samples, n_electrodes)
        '''

        def butter_bandpass_filter(data, lowcut, highcut, sf, order):
            nyq = 0.5 * sf
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            y = signal.filtfilt(b, a, data)
            return y
        
        for electrode_ix in range(data.shape[1]):
            data[:, electrode_ix] = butter_bandpass_filter(data[:,electrode_ix], self.lowcut, self.highcut, self.sf, self.order)

        return data

    def laplacian_filtering(self, data):
        '''Apply laplacian filter to data with neighbor distance as 2 (next next one).
        
        Parameters
        ----------
        data: 2-d array with shape (n_samples, n_electrodes)

        Returns
        -------
        data: 2-d array with shape (n_samples, n_electrodes)
        '''

        # Get labels and coordinates of all channels for cap type
        ch_names, coords = self.get_electrode_position()
        GRIDSHAPE = (max([coord[1] for coord in coords])+1, max([coord[0] for coord in coords])+1)
        
        # Drop assigned channels
        ch_names, coords = zip(*[(ch_name, coord) for ch_name, coord in zip(ch_names, coords) if ch_name not in self.ch_to_drop])

        # Fill in each electrode (by order) into grid
        inds_grid = np.empty(GRIDSHAPE, dtype='int') * np.nan
        for i, ind in enumerate(coords):
            inds_grid[ind[1], ind[0]] = i

        # List neighboring electrodes for each electrode in four directions (with distance 2)
        neighbors = []
        for i, ind in enumerate(coords):
            iy, ix = ind
            neighbors_i = []
            if ix > 1 and ~np.isnan(inds_grid[ix-2, iy]):
                neighbors_i.append(int(inds_grid[ix-2, iy]))
            if ix < GRIDSHAPE[0]-2 and ~np.isnan(inds_grid[ix+2, iy]):
                neighbors_i.append(int(inds_grid[ix+2, iy]))
            if iy > 1 and ~np.isnan(inds_grid[ix, iy-2]):
                neighbors_i.append(int(inds_grid[ix, iy-2]))
            if iy < GRIDSHAPE[1]-2 and ~np.isnan(inds_grid[ix, iy+2]):
                neighbors_i.append(int(inds_grid[ix, iy+2]))
            neighbors.append(neighbors_i)

        # Create row for each electrode indicating all neighbors
        next_adjacency = np.zeros((len(ch_names), len(ch_names)))
        for i, neighbors_i in enumerate(neighbors):
            next_adjacency[i, neighbors_i] = 1
        D = len(ch_names)

        laplacian_next = np.eye(D) - (next_adjacency / np.maximum(np.sum(next_adjacency, axis=1), 1)).T
        #print("Laplacian applied.")

        return data @ laplacian_next.T

    def normalize_channels(self, data, zero_center=False, skip_samples=2000):
        '''Normalize each channel to have mean 0 and standard deviation 1.

        Used during offline function, utilizes mean and standard deviation of 
        entire dataset at once.

        Parameters
        ----------
        data: 2-d array with shape (n_samples, n_electrodes)
        zero_center: If False, a mean of zero is assumed.
        skip_samples: how many samples to skip when calculating the standard deviation.

        Returns
        -------
        data: 2-d array with shape (n_samples, n_electrodes)
        '''
        if zero_center:
            data = data - np.mean(data, axis=0, keepdims=True)
        if len(data) < skip_samples:
            raise ValueError(f'data of length {len(data)} is too short for skip_samples {skip_samples}')
        std = np.sqrt(np.mean(data[skip_samples:None]**2, axis=0, keepdims=True))
        data = data / std
        
        return data

    def test_if_buffer_not_filled(self, data):
        '''This function checks to see if any of the timesteps in the data it 
        has been passed are all 0s. If so, streaming has just started and this sample
        should not be preprocessed because the buffer has not yet accumulated 
        enough data. Just pass through all data unaltered.
        
        This function is intended to be used as part of closed loop preprocessing.'''
        
        #Test if any ticks contain all 0s, indicating buffers still filling
        if np.any(np.all(data==0, axis=0)):
            return True
        else:
            return False

    def throw_channels(self, data):
        '''Throw out channels we don't need.

        Parameters
        ----------
        data: 2-d array with shape (n_samples, n_electrodes)
            Data from eeg_data['databuffer'].
        '''

        ch_names, _ = self.get_electrode_position()
        ch_index_to_drop = [ch_names.index(ch) for ch in self.ch_to_drop]
        data = np.delete(data, ch_index_to_drop, axis=1)
        return data

    def preprocess(self, data):
        '''Manage the whole preprocessing procedure.

        Parameters
        ----------
        data: 2-d array with shape (n_samples, n_electrodes)
            Data from eeg_data['databuffer'].

        Returns
        -------
        data: 2-d array with shape (n_samples, n_electrodes)
        '''

        # Throw out channels
        data = self.throw_channels(data)

        # Apply filters
        if self.apply_bandpass:                     # bandpass filter
            data = self.bandpass_channels(data)
        data = self.laplacian_filtering(data)       # laplacian filter

        # Normalize channels
        #If running offline, utilize mean and std dev of all data to normalize
        if self.online_status == 'offline':
            data = self.normalize_channels(data, zero_center=self.zero_center)
        #If online, normalize with data collected up to this point in time
        if self.online_status == 'online':
            #Check if data buffer not yet filled. If not, return data unaltered
            if self.test_if_buffer_not_filled(data):
                return data
            
            #Instantiate normalization object if needed
            if self.first_run:
                if self.normalizer_type == 'welfords':
                    self.normalizer = Welfords(data, update_mean=self.zero_center) # don't update the mean if not zero-centering
                elif self.normalizer_type == 'running_mean':
                    self.normalizer = Running_Mean(data)
                else:
                    raise ValueError("no such noramlizer as:", self.normalizer)
                self.first_run == False
            else:
                self.normalizer(data)
            #Normalize this sample
            data = ((data-self.normalizer.mean) / self.normalizer.std)

        return data

class Welfords:
    """
    Welford's algorithm computes the standard deviation, mean (and variance) incrementally.
    
    At initializations and future inclusion expects to receive an array of shape [samples, electrodes]
    
    Pass additional samples or batches of samples in by using include.
    
    At any point use .std or .mean to get the mean and standard deviation for everything included so far.
    
    Will return an array of size (electrodes,) for the mean and standard deviation, equivalent to np.mean(all_data, axis=0)
    and np.std(all_data, axis=0)
    """

    def __init__(self, iterable, ddof=1, update_mean=False):
        self.size = iterable.shape[1]
        self.ddof = np.full([self.size,], ddof)
        self.n = 0
        self.mean = np.zeros([self.size,])
        self.M2 = np.zeros([self.size,])
        self.update_mean = update_mean
        self.include(iterable)

    def include(self, datum):
        if datum.ndim == 1:
            self.n += 1
            self.delta = datum - self.mean
            if self.update_mean:
                self.mean += self.delta / self.n
            self.M2 += self.delta * (datum - self.mean)
        
        if datum.ndim == 2:
            for i in range(datum.shape[0]):
                self.n += 1
                self.delta = datum[i, :] - self.mean
                if self.update_mean:
                    self.mean += self.delta / self.n
                self.M2 += self.delta * (datum[i, :] - self.mean)

    @property
    def variance(self):
        return self.M2 / (np.full([self.size,], self.n) - self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)


class Running_Mean:
    '''Class to calculate the running mean and standard deviation of streamed 
    data. It advantages more recent samples over those those are older, governed 
    by the momentum setting (lower momentum = faster decay of old samples).
    
    Takes in a numpy array of data in shape [samples, n_electrodes]. 
    
    After initialization, use .include(data) to feed in more samples.

    All chunks of data passed into the Running_Mean object should be the same 
    size in order for it to properly decay older samples. E.g., if latency is 1 
    second at 1000Hz, all data passed into the object should be [1000, n_electrodes]
    
    Use .mean and .std to get the running mean and standard deviation as 
    calculated at that point in time with all the data fed in included.'''

    def __init__(self, iterable, momentum=0.999):
        self.mean = np.mean(iterable, 0)
        self.std = np.std(iterable, 0)
        self.momentum = momentum

    def include(self, data):
        mean = np.mean(data, 0)
        std = np.std(data, 0)
        self.mean = (self.mean * self.momentum) + ((1-self.momentum) * mean)
        self.std = (self.std * self.momentum) + ((1-self.momentum) * std)


if __name__ == "__main__":

    # put in yaml file name as input (i.e config.yaml)
    yaml_file = sys.argv[1]
    config = utils.read_config(yaml_file)

    # Preprocessing example
    data_file = '/data/raspy/2023-07-22_S1_OL_1_RL/eeg.bin'
    eeg_data = utils.read_data_file_to_dict(data_file)
    preprocessor = DataPreprocessor(config['data_preprocessor'])
    preprocessed_data = preprocessor.preprocess(eeg_data['databuffer'])    # preprocess

    print("preprocessed",preprocessed_data.shape)
