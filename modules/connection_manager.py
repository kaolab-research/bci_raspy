import socket
import numpy as np

LENGTH_LENGTH = 4

class ConnectionManager():
    '''Manager for a single TCP connection'''
    def __init__(self, ip=None, port=None, mode='server', shape=None):
        self.ip = ip
        self.port = port
        self.mode = mode # server or client
        self.buffer = bytes(0) # for reading...?
        self.msg_len = None
        self.shape = shape if shape is not None else (-1) # shape of objects, excluding the first dimension. Use (-1) to fill in a dimension.
        if self.mode == 'server':
            self.server = socket.socket(family = socket.AF_INET, type = socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Set server to be able to reuse the address.
            self.server.bind((self.ip, self.port))
            self.server.listen(1)
            self.server.setblocking(False)
            self.conn = None
        elif self.mode == 'client':
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.conn = None
            try:
                self.client.connect((self.ip, self.port))
                self.conn = self.client
            except ConnectionRefusedError:
                pass
        return
    def sendall(self, msg):
        if self.mode == 'server':
            if self.conn is None:
                try:
                    self.conn, self.addr = self.server.accept()
                    self.conn.setblocking(False)
                    print("Connection established!")
                except:
                    return False
            else:
                try:
                    self.conn.sendall(msg)
                except Exception as e:
                    return False
        elif self.mode == 'client':
            if self.conn is None:
                try:
                    self.client.connect((self.ip, self.port))
                    self.conn = self.client
                    print("Connection established!")
                except ConnectionRefusedError:
                    return False
            if self.conn is not None:
                self.conn.sendall(msg)
            else:
                return False
        return True
    def recv(self, buffer_msg=True):
        if self.conn is None:
            if self.mode == 'server':
                try:
                    self.conn, self.addr = self.server.accept()
                    self.conn.setblocking(False)
                    print("Connection established!")
                except:
                    return False
            elif self.mode == 'client':
                try:
                    self.client.connect((self.ip, self.port))
                    self.conn = self.client
                    print("Connection established!")
                except ConnectionRefusedError:
                    return False
        msg = bytes(0)
        try:
            msg = self.conn.recv(2**24)
        except:
            return False
        if buffer_msg:
            self.buffer = self.buffer + msg
        return msg
    def generate_header(self, msg):
        header = int.to_bytes(len(msg), LENGTH_LENGTH, 'little')
        return header
    def parse_buffer(self, shape=None):
        if shape is None:
            shape = self.shape
        msgs = []
        while True:
            if self.msg_len is None:
                if len(self.buffer) >= LENGTH_LENGTH:
                    self.msg_len = int.from_bytes(self.buffer[0:LENGTH_LENGTH], 'little')
                    self.buffer = self.buffer[LENGTH_LENGTH:None]
                else:
                    break
            if self.msg_len is not None:
                if len(self.buffer) >= self.msg_len:
                    msg = self.buffer[0:self.msg_len]
                    self.buffer = self.buffer[self.msg_len:None]
                    msg_decoded = self.decode_msg(msg).reshape(shape)
                    msgs.append(msg_decoded)
                    self.msg_len = None
                else:
                    break
            else:
                break # redundant but it's okay.
        return msgs
    def encode_arr(self, arr, dtype=np.float64, header=True):
        out = arr.astype(dtype).tobytes()
        if header:
            out = self.generate_header(out) + out
        return out
    def decode_msg(self, msg, dtype=np.float64):
        out = np.frombuffer(msg, dtype=dtype)
        return out