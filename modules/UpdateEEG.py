'''This module will grab EEG data and load it into a buffer'''

if conn is None:
    try:
        conn, addr = serverSocket.accept()
        conn.setblocking(False)
        print("EEG connection established!")
    except:
        pass
else:
    try:
        inputBuffer += conn.recv(bufferSize)
    except Exception as e:
        pass
# Grab data sample from EEG buffer
nSamples = len(inputBuffer) // sizeOfRecord
totalSamples += nSamples
#print(nSamples)
if totalSamples < samplesToSkip: # might skip samplesToSkip +/- a few. doesn't matter too much.
    inputBuffer = inputBuffer[nSamples*sizeOfRecord:] # remove skipped data
    nSamples = 0


if nSamples > 0:
    data = np.fromstring(inputBuffer[:nSamples*sizeOfRecord], dtype=unpackArgs).reshape((-1, nChannels))
    for idx in range(data.shape[0]):
        # directly modify the signal to save time
        # to access, use eegbuffersignal[eegbufferindex - (number_of_samples-1):eegbufferindex+1]
        eegbuffersignal[bufferInd] = data[idx]
        eegbuffersignal[bufferInd - bufferoffset] = data[idx]
        bufferInd = (bufferInd - bufferoffset + 1) % bufferoffset + bufferoffset
        if data[idx][-1] != prevCount + 1 and prevCount > 0:
            print("!!!!!!!!! Missing data between samples {} and {}".format(prevCount, data[idx][-1]))
        prevCount = data[idx][-1]
        #print(bufferInd)
    inputBuffer = inputBuffer[nSamples*sizeOfRecord:] # remove processed data

numEEGSamples[:] = nSamples
eegbufferindex[:] = bufferInd
# is NOT the same as totalSamples, but rather the total samples
# after surpassing samplesToSkip
totalValidEEGSamples += nSamples

t0 = time.time()
tickNo += 1