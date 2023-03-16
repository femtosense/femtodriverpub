"""ZynqPlugin : IOPlugin, helps SPURunner talk to the ZCU104 board

Generally just handles the nuts and bolts of packing raw data into the
ethernet format used when talking to the ZCU104 from the PC

Whether a real-chip SPU is to be driven or the SPU design in FPGA fabric
(direct AXI interface for FPGA vs AXI-over-SPI for real chip or full chip RTL)
is to be driven is encoded at Zynq-firmware level. This driver does not need to change.
"""

import numpy as np
import subprocess
import time
import os

import matplotlib.pyplot as plt

import socket # for real cable run
import femtodriverpub.util.fake_socket as fake_socket # for board-less debug

from femtodriverpub import cfg

from typing import *
from femtodriverpub.typing_help import *

import logging
logger = logging.getLogger(__name__)

# this is the version of the SPU for the real time demo
# AXI-stream for input and output streaming (effectively the rtr up/dn)
#  actually the rtr_ifc layer is stripped
#  talk directly to ifid_pc and RI on downstream side
#    first word is PC, subsequent words are raw data
#    last signifies end of transmission
#  unpacked RO on upstream side
#    first word is PC (mailbox), subsequent words are raw data
#    last signifies end of transmission
# APB-over-AXI for programming mem_conf
# no other registers (including reset)

# eth constants
HOST = "192.168.1.8"
MAXSPEED = None # Mbps, allow for throttling transmission
PORT = 7
MAX_CHUNKSIZE = 512 # bytes to use in send-recv transmission
ENV_RECV_WORDS = 128 # FIXME, size of upstream vector to recv

B_CODESHIFT = 28

MSG_CODES = {
    'apb_read'               : 0,
    'apb_write'              : 1,
    'axis_read'              : 2,
    'axis_write'             : 3,
    'apb_read_reply'         : 4,
    'axis_read_reply'        : 5,
    'axis_read_reply_over'   : 6,
    'spu_top_read'           : 7,
    'spu_top_write'          : 8,  # not implemented
    'host_read'              : 9,  # not implemented
    'host_write'             : 10, # not implemented
    'spu_top_read_reply'     : 11, # not implemented
    'host_read_reply'        : 12, # not implemented
    None : None,
}

def as_32b_hex(val):
    return "0x{:08x}".format(val)

def pretty_print_eth_bytes(h):
        bs = [h[i*2 : (i + 1)*2] for i in range(int(np.ceil(len(h) / 2)))]
        i = 0
        bprint = []
        for b in bs:
            bprint.append(b)
            i += 1
            if i == 4:
                logger.debug("  %s", str(bprint[::-1]))
                bprint = []
                i = 0

class ZynqPlugin:

    def __init__(self, 
            fake_connection=False, 
            fake_hw_recv_vals : ARRAYINT=None):
        """ZynqPlugin is used by HWRunner to send data to and from the board

        provides 
            setup()
            teardown()
            hw_send()
            hw_recv()
            recording-related functions, e.g. start_apb_recording()

        Args:
            fake : bool (default False) :
                don't attempt to connect to actual ethernet stack, just dump traffic into a hole
                useful for generating programming streams without a board
            fake_hw_recv_vals : ARRAYINT (default None)
                values to return from hw_recv with a fake socket
                used for board-less unit tests
        """
        self.fake_socket = fake_connection
        self.fake_hw_recv_vals = fake_hw_recv_vals

        # FIXME, move the recording machinery out of here, up to HWRunner
        self.apb_transaction_records = {};
        self.curr_apb_record = None

        self.setup()

    def __del__(self):
        self.teardown()

    def setup(self):
        """called whenever connection to hardware is established
        In this case, connects the socket
        """
        # connect ethernet
        self._eth_connect()

    def teardown(self):
        """called whenever connection to hardware broken
        In this case, disconnects the socket
        """
        self._eth_disconnect()

    def reset(self):
        """can do platform-specific reset actions here"""
        pass

    @staticmethod
    def _addr_range(msgtype, start_addr, end_addr, length):
        if msgtype == 'axis':
            # zynq RTR addrs are funny
            # the firmware is using addr to signal the last entry
            # we are supposed to set all zeros then 1 for 64b word
            # a previous driver version always wrote the address of the
            # "second" 32b word in a two-word pair as addr1 + 4
            # so the driver is looking for 1, 5 as the signal for terminal word
            assert start_addr == end_addr == 0
            return np.array([0] * (length - 2) + [1, 5], dtype=np.uint32)

        assert length > 0
        assert end_addr != start_addr
        assert (end_addr - start_addr) % length == 0
        increment = (end_addr - start_addr) // length
        #print(start_addr, end_addr, length, increment)
        return np.arange(start_addr, end_addr, increment, dtype=np.uint32)

    def hw_send(self,
            msgtype    : IOTARGET,
            start_addr : int,
            end_addr   : int,
            length     : int,
            vals       : ARRAYU32,
            flush      : bool=True):
        """send raw 32b data to the hardware"""
        # target is (cidx, object, offset)

        assert flush # should be fast enough to not need to queue transactions
        if not isinstance(vals, np.ndarray):
            vals = np.atleast_1d(vals)

        addrs = self._addr_range(msgtype, start_addr, end_addr, length);

        msgbuf = self._eth_pack(msgtype + '_write', addrs, vals, do_print=False)

        if msgtype == 'axis':
            logger.debug("THIS IS WHAT hw_send() SENT")
            h = msgbuf.hex()
            logger.debug(str(h))
            pretty_print_eth_bytes(h)

        curr_idx = 0

        # allow for throttling transmission speed
        if MAXSPEED is not None:
            bits_per_send = MAX_CHUNKSIZE * 8
            T_per_send = bits_per_send / (MAXSPEED * 1e6)

        while curr_idx < len(msgbuf):
            bytes_left = len(msgbuf) - curr_idx
            this_chunk_size = min(bytes_left, MAX_CHUNKSIZE)

            chunk_down = msgbuf[curr_idx : curr_idx + this_chunk_size]
            self.sock.sendall(chunk_down)

            if MAXSPEED is not None:
                time.sleep(T_per_send)

            curr_idx += this_chunk_size

    def hw_recv(self,
            msgtype    : IOTARGET,
            start_addr : int,
            end_addr   : int,
            length     : int) -> ARRAYU32:
        """get raw data back from the hardware"""

        # fake a well-formed reply
        # the fake socket will return something, the unpacker will
        # complain about it being malformed
        # to make this work with most send/recv type networks,
        #  make the first word the mailbox id, usually 0
        # XXX this might not work if multiple messages are expected back
        addrs = self._addr_range(msgtype, start_addr, end_addr, length);

        if self.fake_socket:
            if self.fake_hw_recv_vals is None:
                num_words = len(addrs)
                if msgtype == 'axis':
                    num_words += 2

                fake_data = np.ones((num_words,), dtype=np.uint32)

                fake_data[:] = 0xfedcba9876543210
                if msgtype == 'axis':
                    # fake 0 mailbox id, 'out' route is maxval=1f (it's ignored, however)
                    fake_data[0] = 0x1f
                    fake_data[1] = 0
            else: 
                fake_data = self.fake_hw_recv_vals
            return [fake_data]

        # send the READ_AXIS message
        if msgtype == 'axis':
            ###########################
            # send the AXIS request
            # in this case, the zynq firmware is actually expecting
            # the number of 64B words to ask for
            # so divide by two and add one for the route words
            assert length % 2 == 0
            length_words = np.array([length // 2 + 1], dtype=np.uint32) # +2 32b words, need the route + mailbox id
            null_data = np.zeros((1,), dtype=np.uint32)
            msgbuf = self._eth_pack('axis_read', length_words, null_data, do_print=False)
            assert(len(msgbuf) == 8)
            self.sock.sendall(msgbuf) 
            
            ###########################
            # process the AXIS reply
            # expected size back, including route/PC, also tail code
            # what we get back is 32b words: code, data, code, data
            num_bytes = (len(addrs) + 4) * 8 * 2
            #assert num_bytes < MAX_CHUNKSIZE

            received = self.sock.recv(num_bytes)

            h = received.hex()
            logger.debug("THIS IS WHAT hw_recv() GOT")
            logger.debug(str(h))
            pretty_print_eth_bytes(h)

            unpacked_msgs = self._eth_unpack(received)
            raw_datas = []
            for msg in unpacked_msgs:
                eth_msgtype, raw_data = msg
                # shouldn't get an 'empty' read_reply_over right now
                if msgtype == 'axis':
                    assert eth_msgtype == 'axis_read_reply'
                raw_datas.append(raw_data)
            return raw_datas

        elif msgtype == 'apb':
            # work one chunk at a time: request data, process replies
            null_data = np.zeros_like(addrs)
            msgbuf = self._eth_pack('apb_read', addrs, null_data)
            
            curr_idx = 0

            all_datas = []
            while curr_idx < len(msgbuf):
                bytes_left = len(msgbuf) - curr_idx
                this_chunk_size = min(bytes_left, MAX_CHUNKSIZE)

                chunk_down = msgbuf[curr_idx : curr_idx + this_chunk_size]
                self.sock.sendall(chunk_down)
                
                # expect same-size chunk back
                received = self.sock.recv(this_chunk_size)
                assert len(received) == this_chunk_size
                unpacked_msgs = self._eth_unpack(received)
                assert len(unpacked_msgs) == 1
                _, data = unpacked_msgs[0]
                all_datas.append(data)

                curr_idx += this_chunk_size
            return [np.concatenate(all_datas)]

        elif obj in SYS_REG_ADDRS or obj in CORE_REG_REL_ADDRS:
            # XXX clean this up, make it so you can do more than one reg
            assert len(addrs) == 1
            null_data = np.zeros_like(addrs)
            msgbuf = self._eth_pack('apb_read', addrs, null_data)

            self.sock.sendall(msgbuf) 
            
            # expect same-size chunk back
            received = self.sock.recv(len(msgbuf))
            assert len(received) == len(msgbuf)
            unpacked_msgs = self._eth_unpack(received)
            _, data = unpacked_msgs[0]
            return [data]

        else:
            print('weird target:', target)
            assert False

    # FIXME
    # apb capture stuff should probably be moved to the HWRunner itself
    # can be done at a higher level, this machinery works at the last moment we send an eth message

    def start_apb_recording(self, record_name):
        """Captures APB transactions that have gone out. 
        Used to write programming files for the SD card (e.g. when using the fake_socket)
        """
        self.curr_apb_record = record_name
        self.apb_transaction_records[record_name] = {'addr' : [], 'data' : []}

    def stop_apb_recording(self):
        self.curr_apb_record = None

    def _record_apb(self, addr, data):
        """appends an apb transaction to the record"""
        if self.curr_apb_record is not None:
            self.apb_transaction_records[self.curr_apb_record]['addr'].append(addr)
            self.apb_transaction_records[self.curr_apb_record]['data'].append(data)

    def _hw_addr_map_str(self):
        s = "HW (byte) ADDRESS MAP:\n"
        for (lo, hi), name in ADDR_LOOKUP:
            s += f"{as_32b_hex(lo << 2)} to {as_32b_hex(hi << 2)} : {name}\n"
        return s
        
    def _hw_addr_to_obj_str(self, hw_addr):
        base_addr = hw_addr >> 2
        obj = LOOK_UP_ADDR(base_addr)
        if obj is None:
            return "UNKNOWN ADDR!"
        return obj
    
    def _pretty_addr(self, addr):
        return f"[{as_32b_hex(addr)} = {addr} ({self.io.hw_addr_to_obj_str(addr)})]"

    # FIXME addressing-unpacking stuff should possibly be moved

    def _eth_connect(self):

        if self.fake_socket:
            self.sock = fake_socket.socket(fake_socket.AF_INET, fake_socket.SOCK_STREAM)
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024*1024)

        logger.info("connecting to socket")
        self.sock.connect((HOST, PORT))
        logger.info(" ...success!")

    def _eth_disconnect(self):
        self.sock.close()

    def _record_eth_pack(self, msg_code, addrs, datas):
        # record apb transactions
        if self.curr_apb_record is not None:
            if msg_code == MSG_CODES['apb_write']:
                for addr, data in zip(addrs, datas):
                    self._record_apb(addr, data)

    def _eth_pack(self, msg_codename:str, addrs:ARRAYU32, datas:ARRAYU32, do_print:bool=False) -> ARRAYU32:
        """
        lower index -> higher index
        eth down format is:
          || 28b : addr |  4b code || 4B : 32b data ||
        
        codes:
        
        0 : read APB
          |     0x0 |   4B : addr |   4B : UNUSED |
        
        1 : write APB
          |     0x1 |   4B : addr | 4B : 32b data |
        
        2 : read axi stream
          |     0x2 | 4B : length |   4B : UNUSED |
            sending a single word is enough to get all the data
            if length is not zero, send everything
            else, send only if buffer len == length
        
        3 : write axi stream
          |     0x3 | 4B : UNUSED | 4B : 32b data |
        """
        msg_code = MSG_CODES[msg_codename]

        assert np.all(addrs < (1<<28)) # no code collisions

        stream = np.zeros((len(addrs) * 2,), dtype=np.uint32)

        stream[0::2] = msg_code << B_CODESHIFT
        stream[0::2] = stream[0::2] | addrs
        stream[1::2] = datas

        self._record_eth_pack(msg_code, addrs, datas)

        #print('eth pack half stream')
        #for idx, el in enumerate(stream):
        #    print(idx, ':', hex(el))
        #print(len(stream), 'bytes')
        #print(stream.tobytes())

        return stream.tobytes()

    def _eth_unpack_one_message(self, ints : ARRAYINT) -> Tuple[str, ARRAYU32]:
        """
        strips code word information from single complete ethernet word
        returns tuple with (message_type, raw payload data)

        eth format is:
        lower index -> higher index
          || 28b : addr |  4b code || 4B : 32b data ||
        codes:
        4 : read APB reply
          |     0x4 |   4B : addr | 4B : 32b data |
        
        5 : read AXI stream data
          |     0x5 | 4B : UNUSED | 4B : 32b data |
        
        6 : read AXI stream over
          |     0x6 | 4B : length | 4B : UNUSED   |
            comes after a series of 5s, signifies the final message
            (can be all by itself, if no data or length didn't match)
            the address field reports the length of the buffer
        """
        _get_code = lambda x : x >> B_CODESHIFT
        _get_addr = lambda x : x & ((1 << B_CODESHIFT) - 1)
        first_code = _get_code(ints[0])

        if first_code == MSG_CODES['apb_read_reply']:
            # APB replies
            assert np.all(_get_code(ints[0::2]) == MSG_CODES['apb_read_reply']) 
            # discard code/addrs
            return ('apb_read_reply', ints[1::2])

        elif first_code == MSG_CODES['axis_read_reply_over']:
            return ('axis_read_reply_over', ints[1])

        elif first_code == MSG_CODES['axis_read_reply']:
            body_code = MSG_CODES['axis_read_reply']
            tail_code = MSG_CODES['axis_read_reply_over']

            assert np.all(_get_code(ints[:-2:2]) == body_code) 
            assert        _get_code(   ints[-2]) == tail_code # last should be the "read over"
            msg_ints = ints[:-2]
            return ('axis_read_reply', msg_ints[1::2])
        
        else:
            print(f'GOT A WEIRD CODE BACK {first_code} = {hex(first_code)}')

    def _break_up_messages(self, ints : ARRAYINT) -> List[ARRAYINT]:
        """looks at AXIS code words, breaking single array into a list array at message boundaries for AXIS streams
        APB transactions are always all-in-one
        """
        _get_code = lambda x : x >> B_CODESHIFT

        if _get_code(ints[0]) == MSG_CODES['apb_read_reply']:
            # XXX for now, guaranteed to always be complete, can just take the whole thing
            return [ints]

        # otherwise, could be one or more AXIS outputs
        msgs = []
        idx = 0
        #print(len(ints))
        #print(ints)
        while len(ints) > 0:
            el = _get_code(ints[idx])
                
            if el == MSG_CODES['axis_read_reply_over']:
                msgs.append(ints[idx : idx + 2])
                idx += 2
                ints = ints[idx:]
            elif el == MSG_CODES['axis_read_reply']:
                # find the end
                test_idx = idx
                while _get_code(ints[test_idx]) == MSG_CODES['axis_read_reply']:
                    test_idx += 2
                assert _get_code(ints[test_idx]) == MSG_CODES['axis_read_reply_over']
                msgs.append(ints[idx : test_idx + 2])
                print('found one axis reply message', idx, "to", test_idx)
                idx = test_idx + 2
                ints = ints[idx:]
            else:
                assert False
        
        return msgs

    def _eth_unpack(self, buf:bytes) -> List[ARRAYU32]:
        """
        assumes that buf begins with a control word
        assumes that messages are complete

        """
        ints = np.frombuffer(buf, dtype=np.uint32)
        assert len(ints) % 2 == 0 # need complete words
        
        msgs = self._break_up_messages(ints)
        unpacked = [self._eth_unpack_one_message(msg) for msg in msgs]
        assert len(unpacked) == 1 # XXX might need to fix, don't ever use more than 1 now
        return unpacked


