import numpy as np
import warnings

if __name__ == 'builtins':
    print('buffer_util is not a raspy module...')
    pass

class DoubleCircularBuffer():
    '''
    Utility class for reading and writing to double circular buffers.
    A double circular buffer is two identical circular buffers concatenated.
    This has extra cost on write, but guarantees contiguous read/pull operations (up to the buffer size).
    '''
    # to-do if needed: add a lock, then block on write?
    def __init__(self, np_buffer, np_index, init=True, permissions='all'):
        # np_buffer is an array of shape (2*num_slots, *item_shape)
        # np_index should have shape (1,)
        # init=True sets np_buffer and np_index to their default values in-place
        self.N = np_buffer.shape[0]//2 # max number of unique records
        if init:
            np_buffer[:] = 0
            np_index[:] = self.N
        self.buffer = np_buffer
        self.index = np_index
        self.read_index = self.index.item()
        self.permissions = permissions # 'all', 'write', 'read'
        return
    def write(self, data):
        # writes data to the next available positions in the buffer
        if self.permissions not in ['all', 'write']:
            warnings.warn('Attempted to write to a buffer while in read-only mode!')
            return 0
        if data.ndim == 1:
            data = data.reshape((1, -1))
        n_samples = data.shape[0]
        self.buffer[self.index[0]-self.N:self.index[0]-self.N+n_samples] = data
        wrap_inds = (np.arange(self.index[0], self.index[0]+n_samples) % (2*self.N))
        self.buffer[wrap_inds] = data
        self.index[:] = (self.index - self.N + n_samples) % self.N + self.N
        return n_samples
    def read(self, n_samples=None, copy_data=False):
        # Read everything that hasn't been read yet
        # DOES NOT THROW an error if you've already wrapped around the entire buffer.
        if n_samples is None:
            n_samples = int((self.index - self.read_index) % self.N)
        out = self.pull(n_samples, copy_data=copy_data)
        self.read_index = (self.read_index - self.N + n_samples) % self.N + self.N
        return out
    def pull(self, n_samples, copy_data=False):
        # returns the most recent n_samples records
        # use copy_data if you plan to modify the output in-place
        out = self.buffer[self.index[0]-n_samples:self.index[0]]
        if copy_data:
            out = out.copy()
        return out

class CircularBuffer():
    '''
    Utility class for reading and writing to normal circular buffers
    '''
    def __init__(self, np_buffer, np_index, init=True, permissions='all'):
        # np_buffer is an array of shape (num_slots, *item_shape)
        # np_index should have shape (1,)
        # init=True sets np_buffer and np_index to their default values in-place
        self.N = np_buffer.shape[0] # max number of unique records
        if init:
            np_buffer[:] = 0
            np_index[:] = 0
        self.buffer = np_buffer
        self.index = np_index
        self.permissions = permissions # 'all', 'write', 'read'
        return
    def write(self, data):
        # writes data to the next available positions in the buffer
        if self.permissions not in ['all', 'write']:
            warnings.warn('Attempted to write to a buffer while in read-only mode!')
            return 0
        if data.ndim == 1:
            data = data.reshape((1, -1))
        n_samples = data.shape[0]
        wrap_inds = (np.arange(self.index[0], self.index[0]+n_samples) % (self.N))
        self.buffer[wrap_inds] = data
        self.index[:] = (self.index + n_samples) % self.N
        return n_samples
    def read(self, n_samples=None):
        # Read everything that hasn't been read yet
        # DOES NOT THROW an error if you've already wrapped around the entire buffer.
        if n_samples is None:
            n_samples = int((self.index - self.read_index) % self.N)
        out = self.pull(n_samples)
        self.read_index = (self.read_index + n_samples) % self.N
        return out
    def pull(self, n_samples):
        # returns the most recent n_samples records
        #return self.buffer[self.index[0]-n_samples:self.index[0]]
        wrap_inds = (np.arange(self.index[0]-n_samples, self.index[0]) % (self.N))
        out = self.buffer[wrap_inds]
        return out