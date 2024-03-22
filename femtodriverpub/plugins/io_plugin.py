"""RedisPlugin : IOPlugin, helps SPURunner talk to any redis client

Just handles the packing/unpacking of message data into redis API calls

The client could be anything--initial target is RTL simulator DPI-C
"""

import numpy as np
import subprocess
import time
import os
import io
import yaml
from pathlib import Path
from copy import copy
from abc import abstractmethod

import matplotlib.pyplot as plt

try:
    from hardware_config import cfg
except ImportError:
    from femtodriverpub import cfg  # fall back to 1.2 config (eval systems)

from femtodriverpub.util.hexfile import *
from femtodriverpub.typing_help import *
from typing import *

import redis

import logging

logger = logging.getLogger(__name__)

io_targets = ["apb", "axis", "spu_top", "host"]
CODE_TO_MSG = {i: k for i, k in enumerate(io_targets)}
MSG_TO_CODE = {k: i for i, k in enumerate(io_targets)}


# just makes nicer formatting when emitting yaml
class SpaceDumper(yaml.SafeDumper):
    # insert blank lines between top-level objects
    def write_line_break(self, data=None):
        super().write_line_break(data)
        if len(self.indents) == 1:
            super().write_line_break()


class IORecord:
    def __init__(self, yaml_fname: Optional[str] = None):
        # io_records[record_name][transaction idx][transaction field]
        self.data: dict[list[dict]] = {}

        # load from yaml, if provided
        if yaml_fname is not None:
            with open(yaml_fname, "r") as f:
                self.data = yaml.safe_load(f)

            # go through and convert vals to numpy 32
            for record_name, record in self.data.items():
                for tx in record:
                    if tx["vals"] is not None:
                        tx["vals"] = np.ndarray(tx["vals"], dtype=np.uint32)

    def append(
        self, record, msgtype, direction, start_addr, end_addr, length, vals, comment
    ):
        if record not in self.data:
            self.data[record] = []

        # convert numpy array to list
        if isinstance(vals, np.ndarray):
            vals = [int(v) for v in vals]

        this_record = {
            "msgtype": msgtype,
            "direction": direction,
            "comment": comment,
            "start_addr": start_addr,
            "end_addr": end_addr,
            "length": length,
            "vals": vals,  # will contain multiple things
        }
        self.data[record].append(this_record)

    def filter_msgtypes(self, msgtypes_directions: list[tuple[str]]):
        """return new IORecord, retaining only, (message types, direction) pairs
        can supply 'rw' for direction to get both
        """
        new_record = IORecord()
        for record_name, record in self.data.items():
            for tx in record:
                if (tx["msgtype"], tx["direction"]) in msgtypes_directions or (
                    tx["msgtype"],
                    "rw",
                ) in msgtypes_directions:
                    new_record.append(record_name, **tx)
        return new_record

    @staticmethod
    def _to_hex(vals):
        # use our hex dumper, StringIO lets us dump to memory instead
        s = io.StringIO()
        save_hexfile(s, vals, bits=32)
        hexvals = s.getvalue().splitlines()
        hexvals = [
            "0x" + s for s in hexvals
        ]  # this is really just to get consistent yaml formatting

        if not hasattr(vals, "__iter__"):
            return hexvals[0]

        return hexvals

    def convert_to_hex(self):
        """return new IORecord, but with addresses and data replaced with hex str"""

        new_record = IORecord()
        for record_name, record in self.data.items():
            for tx in record:
                new_tx = copy(tx)
                new_tx["start_addr"] = self._to_hex(new_tx["start_addr"])
                new_tx["end_addr"] = self._to_hex(new_tx["end_addr"])
                if new_tx["vals"] is not None:
                    new_tx["vals"] = self._to_hex(new_tx["vals"])
                new_record.append(record_name, **new_tx)
        return new_record

    def __str__(self):
        return yaml.dump(self.data, sort_keys=False)

    def dump_to_text(self, fname: str, spaces: bool = True, val_comments: bool = True):
        with open(fname, "w") as f:
            f.writelines(self.dump_to_str(spaces, val_comments))

    def dump_to_str(self, spaces: bool = True, val_comments: bool = True) -> str:
        # vals are the only record key to
        # have more than one element/entry
        def _emit_vals(s, k, v):
            if v is not None:
                for idx, val in enumerate(v):
                    if val_comments:
                        # add a comment with each val's addr, for debug
                        addr = tx["start_addr"]
                        if isinstance(addr, str) and addr.startswith("0x"):
                            was_hex = True
                            offset = int(addr, base=16)
                        else:
                            was_hex = False
                            offset = addr
                        offset += idx * 4

                        if was_hex:
                            s += f"{k} {val} #{self._to_hex(offset)}\n"
                        else:
                            s += f"{k} {val} #{offset}\n"
                    else:
                        s += f"{k} {val}\n"

        s = ""
        for record_name, record in self.data.items():
            for tx in record:
                for k, v in tx.items():
                    if k == "vals":
                        _emit_vals(s, k, v)
                    else:
                        s += f"{k} {v}\n"
                if spaces:
                    s += "\n"
        return s

    def dump_to_yaml(self, fname: str, spaces: bool = True):
        """write collected records to files"""

        # print(f'dumping to {fname}')

        if spaces:
            dumper = SpaceDumper
        else:
            dumper = yaml.SafeDumper

        with open(fname, "w") as f:
            yaml.dump(self.data, f, Dumper=dumper, sort_keys=False)

    # legacy, support address-data APB (PROG) format
    def dump_apb_to_text(self, basedir: str):
        apb = self.filter_msgtypes([("apb", "w")])

        # collect addr/data pairs for non-burst APB tx
        addrs = []
        datas = []
        for record_name, record in apb.data.items():
            for tx in record:
                # only record write commands
                if tx["direction"] == "w":
                    curr_addr = tx["start_addr"]
                    for val in tx["vals"]:
                        addrs.append(curr_addr)
                        datas.append(val)
                        curr_addr += 4
                    assert curr_addr == tx["end_addr"]

            save_hexfile(os.path.join(basedir, record_name + "_A"), addrs, bits=32)
            save_hexfile(os.path.join(basedir, record_name + "_D"), datas, bits=32)


class IOPlugin:
    def __init__(self, logfiledir: str = "io_records"):
        """IOPlugins are used by SPURunner to provide low-level IO to the
        SPU or a simulation of one

        provides
            setup()
            teardown()
            hw_send()
            hw_recv()
            recording-related functions, e.g. start_recording()

        Args:
            fake_connection : bool (default False) :
                instantiate fake redis client to subscribe to traffic
            fake_hw_recv_vals : ARRAYINT (default None)
                values to return from hw_recv with a fake client
                used for board-less unit tests
            logfiledir : (default 'io_records') : str
                where to log records

        This parent class specifies the interfaces and implements recording
        """
        # records of traffic sent to SPU
        self.records: IORecord = IORecord()
        self.curr_record = None

        # create log file directory, if it doesn't exist
        self.logfiledir = logfiledir
        if not os.path.exists(self.logfiledir):
            os.makedirs(self.logfiledir)

    @abstractmethod
    def setup():
        """called whenever connection to hardware is established"""
        raise NotImplementedError("derived class must implement")

    @abstractmethod
    def teardown():
        """called whenever connection to hardware broken"""
        raise NotImplementedError("derived class must implement")

    @abstractmethod
    def _hw_send(
        self,
        msgtype: IOTARGET,
        start_addr: int,
        end_addr: int,
        length: int,
        vals: ARRAYU32,
        flush: bool = True,
    ):
        """send burst transactions to the hardware, gets wrapped by recording functions"""
        raise NotImplementedError("derived class must implement")

    @abstractmethod
    def _hw_recv(
        self, msgtype: IOTARGET, start_addr: int, end_addr: int, length: int
    ) -> list[ARRAYU32]:
        """request burst transactions from the hardware, gets wrapped by recording functions"""
        raise NotImplementedError("derived class must implement")

    def hw_send(
        self,
        msgtype: IOTARGET,
        start_addr: int,
        end_addr: int,
        length: int,
        vals: ARRAYU32,
        flush: bool = True,
        comment: Optional[str] = None,
    ):
        """send burst transactions to the hardware

        args:
        -----
        msgtype : IOTARGET = one of ('apb', 'axis', 'host', 'spu_top')
            broadly, which address space this message belongs to
        start_addr : int
            starting byte address of transaction
        end_addr : int
            ending byte address of transaction
        length : int
            length, in 32b words
            note that (end_addr - start_addr) / 4 != length:
            for axis (rtr)  messages, in particular, start_addr = end_addr and the length is the number
            of words to direct at that single address
        vals : ARRAYU32 = numpy array of uint32
            packed data to send
        comment : str
            no effect on function, but shows up in logs
        """
        if self.curr_record is not None:
            self.records.append(
                self.curr_record,
                msgtype,
                "w",
                start_addr,
                end_addr,
                length,
                vals,
                comment,
            )

        self._hw_send(msgtype, start_addr, end_addr, length, vals, flush)

    def hw_recv(
        self,
        msgtype: IOTARGET,
        start_addr: int,
        end_addr: int,
        length: int,
        comment: Optional[str],
    ) -> list[ARRAYU32]:
        """request burst transactions from the hardware

        args:
        -----
        msgtype : IOTARGET = one of ('apb', 'axis', 'host', 'spu_top')
            broadly, which address space this message belongs to
        start_addr : int
            starting byte address of transaction
        end_addr : int
            ending byte address of transaction
        length : int
            length, in 32b words
            note that (end_addr - start_addr) / 4 != length:
            for axis (rtr)  messages, in particular, start_addr = end_addr and the length is the number
            of words to direct at that single address

        returns:
        --------
        vals : ARRAYU32 = numpy array of uint32
            packed data that was received
        """
        to_return = self._hw_recv(msgtype, start_addr, end_addr, length)

        if self.curr_record is not None:
            self.records.append(
                self.curr_record,
                msgtype,
                "r",
                start_addr,
                end_addr,
                length,
                np.concatenate(
                    to_return
                ),  # it's a list of arrays, concatenate them for the record
                comment,
            )

        return to_return

    def replay_record(
        self, record: Union[str, IORecord], record_names: Optional[list[str]] = None
    ):
        """play an IORecord containing only writes into the SPU

        Not tested yet!!!!

        args:
        -----
        record : yaml fname or IORecord object :
            which record to replay
        record_names: list of strs (optional, default None) :
            which record names to play in, None will play them all
        """
        if isinstance(record, str):
            record = IORecord(record)

        for record_name, record in record.data.items():
            if record_names is None or record_name in record_names:
                for tx in record:
                    if tx["direction"] != "w":
                        raise ValueError(
                            "tried to replay a transaction record that has non-writes in it!"
                        )
                    tx_no_direction = copy(tx)
                    tx_no_direction.pop("direction")
                    self.hw_send(**tx_no_direction)

    #########################################
    # recording control functions

    def start_recording(self, record_name: str):
        """mark new separation in recorded streams, starting now"""
        self.curr_record = record_name

    def stop_recording(self):
        """stop recording for now"""
        self.curr_record = None

    def commit_recording(self, fname: str, write_only: bool = False):
        """commit recording to file

        args:
        ----
        write_only : bool, default false
            filter out read requests
        """

        if write_only:
            records = self.filter_msgtypes(
                [(msgtype, "w") for msgtype in get_args(IOTARGET)]
            )
        else:
            records = self.records

        hex_records = records.convert_to_hex()

        full_fname = os.path.join(self.logfiledir, fname)
        records.dump_to_yaml(full_fname)

        hex_fname = Path(fname).stem + "_hex" + Path(full_fname).suffix
        full_hex_fname = os.path.join(self.logfiledir, hex_fname)
        hex_records.dump_to_yaml(full_hex_fname)

        hex_text_fname = Path(fname).stem + "_hex.txt"
        full_hex_text_fname = os.path.join(self.logfiledir, hex_text_fname)
        hex_records.dump_to_text(full_hex_text_fname, spaces=False, val_comments=True)
