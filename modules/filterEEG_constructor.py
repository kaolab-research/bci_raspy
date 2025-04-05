import numpy as np
import scipy.signal


Fs = 1000.0
Fn = [60, 120, 180, 60]
Q = [10, 5, 2, 4]
sosNotch = [np.concatenate(scipy.signal.iirnotch(Fn[i], Q[i], fs=Fs)).reshape((1, 6)) 
    for i in range(len(Fn))]
Fc_lower = 4  # bandpass between 4 and 40Hz
Fc_upper = 40 # bandpass between 4 and 40Hz
wc_lower = Fc_lower/Fs*2.0
wc_upper = Fc_upper/Fs*2.0 # Fc cutoff frequency (2x Fc/Fs = Fc/Nyquist)

# lowpass and highpass butterworth filter
sosLP = scipy.signal.butter(4, wc_upper, btype='lowpass', output='sos')
sosHP = scipy.signal.butter(5, wc_lower, btype='highpass', output='sos')

sos = np.vstack([*sosNotch, sosLP, sosHP])  # add low and high pass filter coefficients
zi0 = scipy.signal.sosfilt_zi(sos)

data_init = np.zeros(66)

zi = None

idx = eegbufferindex[0]
N = eegbuffersignal.shape[0] // 2

