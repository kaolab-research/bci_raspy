import numpy as np
import socket
import time
import argparse
import os
os.system('')

'''
python stream_fake.py --ip 127.0.0.1 --port 7779 --fs 1000.0 --dt 0.001 --std 0.00001
'''

if __name__ != '__main__':
    raise AssertionError('do not import me')

parser = argparse.ArgumentParser()
parser.add_argument('--ip', default='127.0.0.1', type=str, help='Which IP address to send fake streaming data to')
parser.add_argument('--port', default=7779, type=int, help='Which IP address to send fake streaming data to')
parser.add_argument('--fs', default=1000.0, type=float, help='sampling rate of fake data, in Hz.')
parser.add_argument('--dt', default=0.001, type=float, help='how often to send fake data if available, in seconds.')
parser.add_argument('--std', default=0.00001, type=float, help='standard deviation of fake data. data is always clipped between -0.75 and 0.75 to mimic ANT amplifier')
parser.add_argument('--verbose', default=2, type=int, help='verbosity: 2 to print each second sequentially, 1 to print each second, 0 to print nothing.')
args = parser.parse_args()

IP = args.ip
PORT = args.port
Fs = args.fs
if Fs <= 0:
    raise ValueError(f'fs must be positive. Received {args.fs}')
dt = args.dt
if dt <= 0:
    raise ValueError(f'dt must be positive. Received {args.dt}')
std = args.std
if args.verbose not in [0, 1, 2]:
    raise ValueError(f'verbose must be in [0, 1, 2]. Received {args.verbose}')

clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientSocket.connect((IP, PORT))


T = 1/Fs # sampling period
c = 0 # count of total samples
x = np.zeros(66, dtype=np.float64) # buffer for convenience
x[65] = -1 # sample number, initialized to -1 so that first sample has sample number of 0

if args.verbose == 1:
    print('\n', end='')

t0 = time.time()
while True:
    try:
        time.sleep(dt)
        time_since_start = time.time()-t0 # time elapsed since start
        nSamples = int(time_since_start // T) - c # expected number of new samples
        for j in range(nSamples):
            x[0:64] = np.clip(std*np.random.randn(64), -0.75, 0.75)
            x[65] += 1
            msg = x.tobytes()
            clientSocket.sendall(msg)
        c += nSamples
        if int(c/Fs) - int((c-nSamples)/Fs):
            if args.verbose == 2:
                print(str(c))
            elif args.verbose == 1:
                print('\033[F' + str(c))
    except (KeyboardInterrupt, Exception) as e:
        clientSocket.close()
        print('Exception:', e)
        break

