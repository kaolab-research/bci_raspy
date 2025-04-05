import numpy as np
import scipy
import scipy.signal

if __name__ == 'builtins':
    print('data_filter_util is not a raspy module...')
    pass

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

