import numpy as np
#import torch
#import torch.nn as nn
import ast
import pickle
import scipy
import scipy.signal

def load_data(filename, return_dict=True, copy_arr=False):
    with open(filename, 'rb') as openfile:
        name = openfile.readline().decode('utf-8').strip()
        labels = openfile.readline().decode('utf-8').strip()
        dtypes = openfile.readline().decode('utf-8').strip()
        shapes = None
        # shapes can be indicated with a $ to separate.
        if len(dtypes.split('$')) == 2:
            dtypes, shapes = dtypes.split('$')
            dtypes = dtypes.strip()
            shapes = ast.literal_eval(shapes.strip())
        
        labels = labels.split(',')
        dtypes = dtypes.split(',')
        if shapes is None:
            data = np.fromfile(openfile, dtype=[item for item in zip(labels, dtypes)])
        else:
            data = np.fromfile(openfile, dtype=[item for item in zip(labels, dtypes, shapes)])
        if not return_dict:
            return data
        if copy_arr:
            # copy separates the individual arrays from the bulk numpy array, allowing for memory consolidation.
            data_dict = {label: data[label].copy() for label in labels}
        else:
            data_dict = {label: data[label] for label in labels}
        data_dict['name'] = name
        data_dict['labels'] = labels
        data_dict['dtypes'] = dtypes
    return data_dict

def downsample_data(data, keys=[], downsample=1, start=0, end=None, return_dict=True, name=''):
    # keys is a list of keys to downsample and return. None will use all the keys
    # downsample will only keep every {downsample} entries. 1 keeps all.
    # start: index of first entry
    # end: numpy slice end index
    # name: name to assign to out_dict['name'] if this is a dict
    if keys is None:
        if isinstance(data, dict):
            keys = data['labels']
        else:
            keys = data.dtype.names
    
    dtype = []
    for key in keys:
        dtype.append((key, data[key].dtype.str, data[key][0].shape))
    out = np.array(list(zip(*[data[key][start:end:downsample] for key in keys])), dtype=dtype)
    if not return_dict:
        return out
    out_dict = {key: out[key] for key in keys}
    if name is None:
        if isinstance(data, dict) and name in data:
            name = data['name']
        else:
            name = ''
    out_dict['name'] = name
    out_dict['labels'] = keys
    out_dict['dtypes'] = [dt[1] for dt in dtype]
    return out_dict

def resave_data(data, path, name='', labels=None, as_npy=False):
    # name is the name of this data, and goes in the first line of the file.
    # labels is the names of the variables to keep
    # as_npy: whether to save as a .npy format.
    encoding = 'utf-8'
    labels_was_none = (labels is None)

    header = ''
    header += name + '\n'
    if isinstance(data, dict):
        if labels is None:
            labels = data['labels']
        dtypes = [(label, data[label].dtype.str, data[label][0].shape) for label in labels]
        dtypes_str = ','.join([dt[1] for dt in dtypes]) + '$' + ','.join([str(dt[2]) for dt in dtypes])
    else:
        if labels is None:
            labels = data.dtype.names
            dtypes = data.dtype
            dtypes_str = ','.join([data.dtype[i].base.str for i in range(len(data.dtype))]) + '$' + ','.join([str(data.dtype[i].shape) for i in range(len(data.dtype))])
        else:
            dtypes = [(label, data[label].dtype.str, data[label][0].shape) for label in labels]
            dtypes_str = ','.join([dt[1] for dt in dtypes]) + '$' + ','.join([str(dt[2]) for dt in dtypes])

    header += ','.join(labels) + '\n'
    header += dtypes_str + '\n'



    if not labels_was_none or isinstance(data, dict):
        vs = list(zip(*[data[label] for label in labels]))
        data = np.array(vs, dtype=dtypes)
    
    if as_npy:
        np.save(path, data)
    else:
        with open(path, 'wb') as f:
            f.write(header.encode(encoding))
            f.write(data.tobytes())
    return

def load_pickled_data(f):
    out = []
    if isinstance(f, str):
        f = open(f, 'rb')
    while True:
        try:
            out.append(pickle.load(f))
        except:
            break
    f.close()
    return out

class DataFilter():
    '''
    Class for online and offline causal linear filtering using notch and butterworth filters.
    '''
    def __init__(self, fn=[], q=[], fc=None, btype='lowpass', order=1, fs=1000.0):
        self.fn = fn # notch filter frequencies
        self.q = q # notch filter q factors
        self.fc = fc # butterworth filter cutoff frequencies (in Hz)
        self.btype = btype # 'lowpass', 'highpass', or 'bandpass'
        self.order = order # butterworth filter order, or orders if separate highpass (first) and lowpass (second) filters
        self.fs = fs # sampling frequency
        
        self.sosnotch = [np.concatenate(scipy.signal.iirnotch(fn_i, q_i, fs=fs)).reshape((1, 6)) for fn_i, q_i in zip(fn, q)]
        if fc is None:
            self.sosbutter = np.zeros((0, 6))
            self.sos = np.vstack([*self.sosnotch, self.sosbutter])
        elif (isinstance(fc, int) or isinstance(fc, float)) and btype != 'bandpass':
            wc = fc/fs*2.0
            self.sosbutter = scipy.signal.butter(order, wc, btype=btype, output='sos')
            self.sos = np.vstack([*self.sosnotch, self.sosbutter])
        elif len(fc) == 2 and btype == 'bandpass':
            wc = [fc[0]/fs*2.0, fc[1]/fs*2.0]
            if isinstance(order, int):
                self.sosbutter = scipy.signal.butter(order, wc, btype=btype, output='sos')
                self.sos = np.vstack([*self.sosnotch, self.sosbutter])
            else:
                self.soshp = scipy.signal.butter(order[0], wc[0], btype='highpass', output='sos')
                self.soslp = scipy.signal.butter(order[1], wc[1], btype='lowpass', output='sos')
                self.sos = np.vstack([*self.sosnotch, self.soslp, self.soshp])
        else:
            raise ValueError('fc must match btype')
        
        self.zi0 = scipy.signal.sosfilt_zi(self.sos)
        self.zi = None
        return
    def filter_data(self, data):
        # data should have shape (time, channels)
        if data.ndim == 1:
            zi = self.zi0*data[0]
            out, zo = scipy.signal.sosfilt(self.sos, data, zi=zi)
        else:
            zi = (self.zi0[..., None]@data[0].reshape((1, -1)))
            out, zo = scipy.signal.sosfilt(self.sos, data, axis=0, zi=zi)
        return out
    def reset_state(self):
        self.zi = None
        return
    def filter_online(self, data):
        # data should have shape (time, channels)
        if data.ndim == 1:
            if self.zi is None:
                self.zi = self.zi0*data[0]
            out, self.zi = scipy.signal.sosfilt(self.sos, data, zi=self.zi)
        else:
            if self.zi is None:
                self.zi = (self.zi0[..., None]@data[0].reshape((1, -1)))
            out, self.zi = scipy.signal.sosfilt(self.sos, data, axis=0, zi=self.zi)
        return out

'''
def slidingwindow(data, width, stride=1, dilation=1, batch_dim=False):
    # data is of shape (batch, length, *data_dims) if batch_dim=True
    # data is of shape (length, *data_dims) if batch_dim=False
    # stride is distance between starting points of consecutive windows
    # dilation is distance between consecutive samples in a given window
    if len(data.shape) == 1:
        data = data[:, None]
    windowwidth = (width-1)*dilation+1
    
    if batch_dim:
        Ndata = len(range(windowwidth-1, data.shape[1], stride))
        shape = (data.shape[0], Ndata, width, *data.shape[2:None])
    else:
        Ndata = len(range(windowwidth-1, data.shape[0], stride))
        shape=(Ndata, width, *data.shape[1:None])
    
    if isinstance(data, np.ndarray):
        if batch_dim:
            strides = (data.itemsize*np.product(data.shape[1:None]), stride*data.strides[1], 
                       dilation*data.itemsize*np.product(data.shape[2:None]), *data.strides[2:None])
        else:
            strides = (stride*data.strides[0], dilation*data.itemsize*np.product(data.shape[1:None]), *data.strides[1:None])
        return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    if isinstance(data, torch.Tensor):
        if batch_dim:
            strides = (np.product(data.shape[1:None]), stride*data.stride()[1], 
                       dilation*np.product(data.shape[2:None]), *data.stride()[2:None])
        else:
            strides = (stride*data.stride()[0], dilation*np.product(data.shape[1:None]), *data.stride()[1:None])
        return torch.as_strided(data, size=shape, stride=strides)
    raise ValueError('data must be a numpy array or a torch Tensor')
    return

def subsequences_lims(length, width=1, dilation=1, position=-1):
    if position < 0:
        offset = (width+position)*dilation
    else:
        offset = position*dilation
    # out[0] <= inds < out[1], left-inclusive, right-exclusive
    return (offset, length - (width-1-position)*dilation)

def subsequences(data, inds, width=1, dilation=1, position=-1):
    # returns windows of width width and spread (width-1)*dilation + 1 such that out[:, position, :] = data[inds]
    # Should have -((width-1)*dilation+1) <= position < (width-1)*dilation+1
    if position < 0:
        if position < -width:
            raise ValueError('position is less than -width.')
        offset = (width+position)*dilation
    else:
        if position >= width:
            raise ValueError('position is greater than or equal to the width. (0-indexing)')
        offset = position*dilation
    if isinstance(inds, np.ndarray):
        if np.issubdtype(inds.dtype, np.bool):
            if len(inds) != len(data):
                raise ValueError('Boolean indexes of length {} mismatched with data of length {}'.format(len(inds), len(data)))
            inds = inds.nonzero()[0]
    if len(inds) > 0:
        if np.min(inds) - offset < 0:
            raise ValueError('Minimum index is less than the offset {}. Trying to access data of index less than 0.'.format(offset))
        if position >= 0:
            if np.max(inds) + (width-1-position)*dilation >= data.shape[0]:
                raise ValueError('Maximum index {} plus (width-1-position)*dilation {} is greater than or equal to length {} of data.'.format(
                    np.max(inds), (width-1-position)*dilation, data.shape[0]))
    slidingdata = slidingwindow(data, width, dilation=dilation)
    return slidingdata[np.asarray(inds)-offset]
'''
