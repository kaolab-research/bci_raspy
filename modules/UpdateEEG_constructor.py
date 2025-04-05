import socket
import numpy as np

print('#'*50)
print('INITIALIZING EEG...')

nChannels = 66 # number of eeg channels (same for 32 and 64)
sizeOfFloat = 4
sizeOfDouble = 8
sizeOfRecord = nChannels * sizeOfDouble

inputBuffer = bytes(0)
unpackArgs = "<" + str(nChannels) + 'd' # how to interpret data in the inputBuffer
data = np.zeros((1, nChannels))

#bufferoffset = 500 # >= no. of contiguous samples needed. #Must be at most half of eegbuffersignal's first dimension
bufferoffset = eegbuffersignal.shape[0] // 2
bufferInd = bufferoffset
eegbufferindex[:] = bufferInd

prevCount = 0 # used to test for missing data
totalSamples = 0
samplesToSkip = 0 # number of samples to skip because of filtering.

# socket setup
#IP = '192.168.1.45'
#PORT = 7779
IP = params['IP']
PORT = params['PORT']
#print('i am here')
bufferSize = 2 ** 30 # should be a power of 2 and large enough to read at least 2000 hz * sizeOfRecord * dt bytes

serverSocket = socket.socket(family = socket.AF_INET, type = socket.SOCK_STREAM)
serverSocket.bind((IP, PORT))
serverSocket.listen(1)
serverSocket.setblocking(False)
conn = None
print ('EEG socket ready!')

#serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1000000)
#print(serverSocket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))

totalValidEEGSamples[:] = 0

import time
t0 = time.time()
tickNo = 0