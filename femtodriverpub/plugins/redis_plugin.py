"""RedisPlugin : IOPlugin, helps SPURunner talk to any redis client

Just handles the packing/unpacking of message data into redis API calls

The client could be anything--initial target is RTL simulator DPI-C
"""

import numpy as np
import subprocess
import time
import os

import matplotlib.pyplot as plt

from femtodriverpub import cfg

from typing import *
from femtodriverpub.typing_help import *

io_targets = ['apb', 'axis', 'host', 'spu_top']
CODE_TO_MSG = {i : k for i, k in enumerate(io_targets)}
MSG_TO_CODE = {k : i for i, k in enumerate(io_targets)}

import redis

import logging
logger = logging.getLogger(__name__)

# originally were trying to put everything into one addr space
# but now we pack the message type into the message
# easier to sort things that way than by looking at addr

# lower half of address space used for SPU APB
# upper half is system regs
SYS_ADDR0 = 2**31
NUM_SPI_REG = 16*1024 # don't need this many, but reserved
NUM_HOST_REG = 16*1024  # don't need this many, but reserved

ADDR_SPACE_SIZES = {
    'apb'    : 2**31,
    'axis'   : 4,
    'spi'    : NUM_SPI_REG*4,
    'host'   : NUM_HOST_REG*4,
}
cum_sizes = np.cumsum([v for v in ADDR_SPACE_SIZES.values()])
ADDR_SPACE = {k : v for k, v in zip(ADDR_SPACE_SIZES.keys(), cum_sizes)}

#def redis_addr_map(msgtype, offset):
#    foo = ADDR_SPACE[msgtype]
#    return ADDR_SPACE[msgtype] + offset

def redis_addr_map(msgtype, offset):
    return offset

def as_32b_hex(val):
    return "0x{:08x}".format(val)

class RedisPlugin:
    def __init__(self, 
            fake_connection=False, 
            fake_hw_recv_vals:ARRAYINT=None):
        """RedisPlugin is used by HWRunner to send data to and from another redis client
        that "implements" wraps the SPU (or a simulation of the SPU)

        provides 
            setup()
            teardown()
            hw_send()
            hw_recv()
            recording-related functions, e.g. start_apb_recording()

        Args:
            fake_connection : bool (default False) :
                instantiate fake redis client to subscribe to traffic
            fake_hw_recv_vals : ARRAYINT (default None)
                values to return from hw_recv with a fake client
                used for board-less unit tests

        we'll use redis with the following queues

        (32b data for all elements)

        req : (read)
            [0, (=read)
             target_type_code, 
             base_addr, 
             final_addr, 
             len]

        req : (write)
            [1, (=write)
             target_type_code, 
             base_addr, 
             final_addr, 
             len,
             data, data, data, ...]

        reply (filled by receiver) : 
            [data, data, data, ...]


        """
        if fake_connection:
            self.start_fake_client(fake_hw_recv_vals)
        self.fake_client_proc = None

        # FIXME, move the recording machinery out of here, up to SPURunner?
        self.apb_transaction_records = {};
        self.curr_apb_record = None

        self.setup()

    def start_fake_client(self, fake_hw_recv_vals):
        # save the fake vals
        valfile_fname = 'fake_valfile.txt'
        np.savetxt(valfile_fname, fake_hw_recv_vals)
        this_dir_path = os.path.dirname(os.path.realpath(__file__))
        fake_client_script = os.path.join(this_dir_path, 'fake_redis_client.py')
        self.fake_client_proc = subprocess.Popen([f'python {fake_client_script} --valfile={valfile_fname}'], shell=True)
        logger.info(f"fake client started with PID {self.fake_client_proc.pid}!")
        # can be killed by setting key 'client_shutdown'

    def _init_redis(self):
        self.r = redis.Redis()
        self.r.set('client_shutdown', 0)
        self.r.set('client_pause', 1)
        self.r.set('client_running', 'none')
        self.r.delete('req')
        self.r.delete('reply')

    def __del__(self):
        self.teardown()

    def reset(self):
        self.r.set('client_pause', 0)

    def setup(self):
        """called whenever connection to hardware is established
        In this case, connects the socket
        """
        self._init_redis()

    def teardown(self):
        """called whenever connection to hardware broken
        In this case, disconnects the socket
        """
        # will cause subprocess to exit
        self.r.set('client_shutdown', 1)
        # wait for it to exit
        if self.fake_client_proc is not None:
            self.fake_client_proc.wait()
        self.r.set('client_pause', 1)


    def _push_redis(self, queue_name, vals):
        """low-level redis queue push"""
        assert queue_name in ['req']
        for val in vals:
            #print('pushing', val, "to", queue_name)
            self.r.rpush(queue_name, int(val))

    def _pop_redis(self, queue_name, num_to_pop):
        assert queue_name in ['reply']
        """low-level redis queue pop"""
        popped = []
        print("waiting for reply")
        while len(popped) < num_to_pop:
            _, val_bytes = self.r.blpop(queue_name)
            val = int(val_bytes)
            popped.append(val)
            print('got', val)
        return np.array(popped, dtype=np.uint32)

    def hw_send(self,
            msgtype    : IOTARGET,
            start_addr : int,
            end_addr   : int,
            length     : int,
            vals       : ARRAYU32,
            flush      : bool=True):
        """send burst transactions to the hardware"""

        if not flush:
            raise NotImplementedError()

        if not isinstance(vals, np.ndarray):
            vals = np.atleast_1d(vals)

        req_data = [
            1, 
            MSG_TO_CODE[msgtype], 
            redis_addr_map(msgtype, start_addr),
            redis_addr_map(msgtype, end_addr),
            length] + list(vals)
        print("write", req_data[0:7])

        self._push_redis('req', req_data)
        print('len is now', self.r.llen('req'))

    def hw_recv(self,
            msgtype    : IOTARGET,
            start_addr : int,
            end_addr   : int,
            length     : int) -> ARRAYU32:
        """recv burst transactions from the hardware"""

        req_data = [
            0, 
            MSG_TO_CODE[msgtype], 
            redis_addr_map(msgtype, start_addr),
            redis_addr_map(msgtype, end_addr),
            length] 
        print('sending request of len', length)
        self._push_redis('req', req_data)
        return [self._pop_redis('reply', length)] # XXX FIXME multiple RTR messages?

    # FIXME
    # apb capture stuff should probably be moved to the HWRunner itself
    # can be done at a higher level, this machinery works at the last moment we send an eth message

    def start_apb_recording(self, record_name):
        """Captures APB transactions that have gone out. 
        Used to write programming files for the SD card (e.g. when using the fake_client)
        """
        return # FIXME
        self.curr_apb_record = record_name
        self.apb_transaction_records[record_name] = {'addr' : [], 'data' : []}

    def stop_apb_recording(self):
        return # FIXME
        self.curr_apb_record = None

    def _record_apb(self, addr, data):
        """appends an apb transaction to the record"""
        return # FIXME
        if self.curr_apb_record is not None:
            self.apb_transaction_records[self.curr_apb_record]['addr'].append(addr)
            self.apb_transaction_records[self.curr_apb_record]['data'].append(data)

