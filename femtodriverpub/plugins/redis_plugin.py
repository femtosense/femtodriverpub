"""RedisPlugin : IOPlugin, helps SPURunner talk to any redis client

Just handles the packing/unpacking of message data into redis API calls

The client could be anything--initial target is RTL simulator DPI-C
"""

import numpy as np
import subprocess
import time
import os

import matplotlib.pyplot as plt

try:
    from hardware_config import cfg
except ImportError:
    from femtodriverpub import cfg  # fall back to 1.2 config (eval systems)

from femtodriverpub.plugins.io_plugin import *

from femtodriverpub.typing_help import *
from typing import *

import redis

import logging

logger = logging.getLogger(__name__)

io_targets = ["apb", "axis", "spu_top", "host"]
CODE_TO_MSG = {i: k for i, k in enumerate(io_targets)}
MSG_TO_CODE = {k: i for i, k in enumerate(io_targets)}


# originally were trying to put everything into one addr space
# but now we pack the message type into the message
# easier to sort things that way than by looking at addr

# lower half of address space used for SPU APB
# upper half is system regs
SYS_ADDR0 = 2**31
NUM_SPI_REG = 16 * 1024  # don't need this many, but reserved
NUM_HOST_REG = 16 * 1024  # don't need this many, but reserved

ADDR_SPACE_SIZES = {
    "apb": 2**31,
    "axis": 4,
    "spi": NUM_SPI_REG * 4,
    "host": NUM_HOST_REG * 4,
}
cum_sizes = np.cumsum([v for v in ADDR_SPACE_SIZES.values()])
ADDR_SPACE = {k: v for k, v in zip(ADDR_SPACE_SIZES.keys(), cum_sizes)}

# def redis_addr_map(msgtype, offset):
#    foo = ADDR_SPACE[msgtype]
#    return ADDR_SPACE[msgtype] + offset


def redis_addr_map(msgtype, offset):
    return offset


def as_32b_hex(val):
    return "0x{:08x}".format(val)


# address map
HOST_REGS = {"WAIT": 0, "HARD_RESET": 1}


class RedisPlugin(IOPlugin):
    def __init__(
        self,
        fake_connection=False,
        no_server_start=False,
        fake_hw_recv_vals: ARRAYINT = None,
        logfiledir: str = "io_records",
    ):
        """RedisPlugin is used by HWRunner to send data to and from another redis client
        that "implements" wraps the SPU (or a simulation of the SPU)

        provides, (partially via parent class IOPlugin)
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
            log_to_file=True

        we'll use redis with the following queues:

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
        if fake_connection and not no_server_start:
            self.start_fake_client(fake_hw_recv_vals)
        elif not no_server_start:
            self.start_redis_server()
        else:
            self.fake_client_proc = None
            self.redis_server_proc = None

        self.setup()
        super().__init__(logfiledir=logfiledir)

    @staticmethod
    def _get_user_port():
        uid = os.getuid()
        # 6379 is redis default, assign each user a port above that
        # this has to match the formula that the redis_client has
        # e.g. femtocad/femtorunners/questa/redis_client.c:_GetUserPort()
        return uid % 20 + 6379

    def start_redis_server(self, port=None):
        portid = self._get_user_port()
        self.redis_server_proc = subprocess.Popen(
            ["redis-server", "--port", f"{portid}"]
        )
        logger.info(
            f"started redis server on port {portid}, pid = {self.redis_server_proc}"
        )

    def start_fake_client(self, fake_hw_recv_vals):
        # opens up a subprocess to run the client in
        this_dir_path = os.path.dirname(os.path.realpath(__file__))
        fake_client_script = os.path.join(this_dir_path, "fake_redis_client.py")

        if fake_hw_recv_vals is not None:
            # save the fake vals
            valfile_fname = "fake_valfile.txt"
            np.savetxt(valfile_fname, fake_hw_recv_vals)
            args = ["python", fake_client_script, "--valfile", valfile_fname]
        else:
            args = ["python", fake_client_script]

        self.fake_client_proc = subprocess.Popen(args)
        self.start_redis_server()
        logger.info(f"fake client started with PID {self.fake_client_proc.pid}!")
        # can be killed by setting key 'client_shutdown'

    def wait_for_redis(self):
        """waits for redis server to come online"""
        while True:
            try:
                self.r.ping()
                break
            except redis.ConnectionError:
                time.sleep(1)
                logger.info("waiting for redis-server to come online")

    def _init_redis(self):
        portid = self._get_user_port()
        self.r = redis.Redis(port=portid)
        self.wait_for_redis()
        self.r.flushdb()
        self.r.set("client_shutdown", 0)
        self.r.set("client_freerunning", 0)
        self.r.delete("req")
        self.r.delete("reply")
        self.set_encrypted(False)

    def set_encrypted(self, encrypted):
        # workaround: encrypted disables fast memory programming
        self.r.set("client_encrypted", int(encrypted))
        self.r.set("client_fast_mem_programming", int(not encrypted))

    def __del__(self):
        self.teardown()

    def reset(self, num_hold_cycles=20):
        """hard reset of SPU, commands host to pulse the reset pad"""

        # set reset to 1
        self.hw_send(
            "host",
            HOST_REGS["HARD_RESET"],
            HOST_REGS["HARD_RESET"] + 1,
            1,
            np.array([1], dtype=np.uint32),
            comment="reset high",
        )

        # hold reset for 20 cycles
        self.wait(num_hold_cycles)

        # set reset to 0
        self.hw_send(
            "host",
            HOST_REGS["HARD_RESET"],
            HOST_REGS["HARD_RESET"] + 1,
            1,
            np.array([0], dtype=np.uint32),
            comment="reset low",
        )

    def setup(self):
        """called whenever connection to hardware is established
        In this case, connects to the redis server
        """
        self._init_redis()

    def teardown(self):
        """called whenever connection to hardware broken
        In this case, disconnects from the redis server
        """
        # will cause subprocess to exit
        try:
            self.r.set("client_shutdown", 1)
            logger.info("told client to exit")
            self.wait_for_process(self.fake_client_proc)
            self.wait_for_process(self.redis_server_proc)
        except AttributeError:
            logger.warning(f"init was not complete yet during teardown")
        except redis.ConnectionError:
            logger.warning(f"redis server already down during teardown")

    def wait_for_process(self, proc, timeout_secs=3):
        if proc is None:
            return

        try:
            logger.debug(f"waiting {timeout_secs} secs for process {proc.pid} to exit")
            proc.wait(timeout=timeout_secs)
        except subprocess.TimeoutExpired:
            logger.debug(f"timeout expired, killing process {proc.pid}")
            proc.kill()

    def wait(self, sim_cycles):
        """tell the simulator to cycle"""
        self.hw_send(
            "host",
            HOST_REGS["WAIT"],
            HOST_REGS["WAIT"] + 1,
            1,
            np.array([sim_cycles], dtype=np.uint32),
            comment=f"cycle reference clock for {sim_cycles} cycles",
        )

    def _push_redis(self, queue_name, vals):
        """low-level redis queue push"""
        assert queue_name in ["req"]
        for val in vals:
            self.r.rpush(queue_name, int(val))

    def _pop_redis(self, queue_name, num_to_pop):
        assert queue_name in ["reply"]
        """low-level redis queue pop"""
        popped = []
        logger.debug("waiting for reply")
        while len(popped) < num_to_pop:
            _, val_bytes = self.r.blpop(queue_name)
            val = int(val_bytes)
            popped.append(val)
        return np.array(popped, dtype=np.uint32)

    def _hw_send(
        self,
        msgtype: IOTARGET,
        start_addr: int,
        end_addr: int,
        length: int,
        vals: ARRAYU32,
        flush: bool = True,
    ):
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
            length,
        ] + list(vals)

        self._push_redis("req", req_data)
        logger.debug(f"pushed to redis, req queue len is now {self.r.llen('req')}")

    def _hw_recv(
        self, msgtype: IOTARGET, start_addr: int, end_addr: int, length: int
    ) -> list[ARRAYU32]:
        """recv burst transactions from the hardware"""

        if msgtype == "axis":
            length += 2  # for header, a little weird how the plugin uses this vs addr

        req_data = [
            0,
            MSG_TO_CODE[msgtype],
            redis_addr_map(msgtype, start_addr),
            redis_addr_map(msgtype, end_addr),
            length,
        ]
        logger.debug(f"sending request of len {length}")
        self._push_redis("req", req_data)
        return [self._pop_redis("reply", length)]  # XXX FIXME multiple RTR messages?
