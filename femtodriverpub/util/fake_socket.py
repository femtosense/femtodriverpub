AF_INET = None
SOCK_STREAM = None

import numpy as np

class socket:
    def __init__(self, foo, bar):
        self.send_data = bytes()

    def connect(self, host_port):
        pass

    def sendall(self, data):
        self.send_data += data

    def recv(self, num_bytes):
        fake_u32 = np.zeros((num_bytes // 4,), dtype=np.uint32)
        # give it an acceptable code
        fake_u32[::3] = 4 # read reply
        fake_bytes = fake_u32.tobytes()
        return fake_bytes

    def close(self):
        pass
        

