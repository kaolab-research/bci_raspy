import numpy as np
import socket

'''
# in signals
  gaze_buffer:
    shape: (600, 32)
    dtype: float64
  gaze_buffer_idx:
    shape: 1
    dtype: int16
# in modules
  recv_gaze_buffer:
    constructor: True
    destructor: True
    sync:
      - timer
    out:
      - gaze_buffer
      - gaze_buffer_idx
    trigger: False
    params:
      #IP: ''
      PORT: 7780
# in logger_disk
          gaze:
            IP: '127.0.0.1'
            PORT: 7702
# in logger
        gaze:
          index: gaze_buffer_idx
          buffers:
            - gaze_buffer
    #....
          gaze:
            IP: '127.0.0.1'
            PORT: 7703
'''

'''
# gaze info:
  # Index ranges for each key.
  0  1  device_time_stamp
  1  2  system_time_stamp
  2  4  left_gaze_point_on_display_area
  4  7  left_gaze_point_in_user_coordinate_system
  7  8  left_gaze_point_validity
  8  9  left_pupil_diameter
  9  10 left_pupil_validity
  10 13 left_gaze_origin_in_user_coordinate_system
  13 16 left_gaze_origin_in_trackbox_coordinate_system
  16 17 left_gaze_origin_validity
  17 19 right_gaze_point_on_display_area
  19 22 right_gaze_point_in_user_coordinate_system
  22 23 right_gaze_point_validity
  23 24 right_pupil_diameter
  24 25 right_pupil_validity
  25 28 right_gaze_origin_in_user_coordinate_system
  28 31 right_gaze_origin_in_trackbox_coordinate_system
  31 32 right_gaze_origin_validity
'''

num_float64 = 32 #gaze_buffer.shape[1] # number of float64 numbers to receive from eye tracking stream (integer). MUST ALWAYS MATCH gaze_buffer's second dimension

IP = socket.gethostbyname(socket.gethostname())
if 'IP' in params:
    IP = params['IP']
print('IP:', IP)
PORT = params['PORT']

buffer = bytes(0)

server = socket.socket(family = socket.AF_INET, type = socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Set server to be able to reuse the address.
server.bind((IP, PORT))
server.listen(1)
server.setblocking(False)
conn = None # actual connection with the gaze data streamer

# dtype is float64, shape (n,) -> 8*n bytes
record_num_bytes = 8*num_float64 # Number of bytes per row. 8 bytes per float64 number

bufferoffset = gaze_buffer.shape[0] // 2
gaze_buffer_idx[:] = bufferoffset # position where next data point will go.