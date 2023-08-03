import femtorun as fr
from femtodriverpub.typing_help import *
from typing import *
from femtodriverpub import cfg

from femtorun import FemtoRunner
from femtodriverpub.plugins.zynq_plugin import ZynqPlugin
from femtodriverpub.plugins.redis_plugin import RedisPlugin

import numpy as np
from collections import defaultdict
import textwrap
import copy

import time
import os

import yaml

import logging
logger = logging.getLogger(__name__)

# import correct address map for this version
if cfg.ISA == 1.2:
    from femtodriverpub.addr_map_spu1p2 import *
else:
    raise NotImplementedError(f"unrecognized ISA version {cfg.ISA}")

# program/dump only used locations in DM/TM
# saves programming time
LIMIT_PROGRAMMING = True

SEND_TO_RECV_WAIT = 1.0 # seconds

def load_hexfile(filename):
    try:
        vals = np.loadtxt(filename, dtype='uint64', usecols=(0,), converters={0: lambda s: int(s, 16)})
    except UserWarning:
        vals = np.array([], dtype=np.uint64)
    return np.atleast_1d(vals)

def save_hexfile(fname, vals, bits=None):

    bits_to_fmt = lambda x : '%0' + str(int(np.ceil(x) / 4)) + 'x'

    def hex_fmt_for_mem(mem):
        membits = {
            'DM'    : cfg.B_DATA_WORD,
            'TM'    : cfg.B_TABLE_WORD,
            'SB'    : cfg.B_SB,
            'RQ'    : cfg.B_RQ,
            'PB'    : cfg.B_PC,
            'IM'    : cfg.B_INSTR,
        }
        return bits_to_fmt(membits[mem])

    #np.savetxt(fname, vals, fmt=hex_fmt_for_mem(mem))
    # XXX should infer memory type
    if bits is None:
        fmt = hex_fmt_for_mem('DM')
    else:
        fmt = bits_to_fmt(bits)
    np.savetxt(fname, np.atleast_1d(vals), fmt=fmt)


class SPURunner(FemtoRunner):

    def __init__(self,
            path_to_memory_images:str,
            platform:str='zcu104',
            fake_connection:bool=False,
            image_fname_pre:str='test',
            **kwargs): # catch unused kwargs used by other runners
        """Driver for SPU hardware
        Args:
            path_to_memory_images (str) : 
                path to the unzipped output of femtocrux,
                which contains memory images to pack into an SD programming file
                (future: or to program over a wired connection) 
                (future: or to generate an EVK SPI flash image for)
            platform : str (default "zcu104") : 
                options (redis, zcu104)
                Hardware platform to target (PCB host board).
                Under the hood, will select a different IOPlugin to use with SPURunner.
                So far, all EVK hosts use the zcu104's SD card format,
                so this can be used to generate SD cards for any host.
                Redis platform connects to a redis server, 
                which can be connected to just about anything.
            fake_connection : bool :
                don't actually connect to the hardware, useful if you just want to create an SD card
            image_fname_pre : str : (default = 'test')
                in the memory image folder, what's the preamble that each file starts with
        """
        fake_connection = True
        fake_hw_recv_vals = None
        self.name = f'spu_runner_{platform}'
        self.image_fname_pre = image_fname_pre
        
        ##########################################################
        # supported hardware platforms -> IO plugins
        # IO plugin classes must provide hw_send(), hw_recv()
        platform_to_plugin = {
            'zcu104' : ZynqPlugin,
            'redis' : RedisPlugin,
        }
        if platform == 'redis':
            raise NotImplementedError("redis runner not yet supported pending future release")

        if platform in platform_to_plugin:
            self.io = platform_to_plugin[platform](fake_connection=fake_connection, fake_hw_recv_vals=fake_hw_recv_vals)
        else: 
            raise NotImplementedError(f"unrecognized hardware platform {platform}, must be one of {platform_to_plugin.keys()}")

        self.data_dir = path_to_memory_images

        self._load_meta_yaml(os.path.join(path_to_memory_images, 'metadata.yaml'))

        super().__init__(None, None)

    def _addr_range(self, obj:str, offset:int, 
            length:int, core:Union[int,None]=None) -> Tuple[int, int, int]: # base, end, len
        """uses address maps to compute byte addresses from an object string and offset
        note that set_var()/get_var(), used to program data, takes a slightly different route
        still word/element addresses, width depends on target
        """
        
        if obj in CONF_ADDRS or obj in CORE_REG_REL_ADDRS or obj in SYS_REG_ADDRS:
            if obj in CONF_ADDRS or obj in CORE_REG_REL_ADDRS:
                assert core is not None
            elif obj in SYS_REG_ADDRS:
                assert core is None

            base_addr = OBJ_TO_BYTE_ADDR(obj, core, offset)
            end_addr = OBJ_TO_BYTE_ADDR(obj, core, offset + length)
            if obj in CONF_ADDRS:
                assert end_addr - base_addr == 8 * length # can't cross a bank boundary in one call
            return base_addr, end_addr, length

        elif obj == 'RTR':
            assert core is None
            base_addr = end_addr = OBJ_TO_BYTE_ADDR(obj)
            return base_addr, end_addr, length

        elif obj in SPI_REGS:
            assert core is None
            base_addr = OBJ_TO_BYTE_ADDR(obj, core, offset)
            end_addr = OBJ_TO_BYTE_ADDR(obj, core, offset + length)
            return base_addr, end_addr, length

        else: 
            # something not supported for this HW
            raise NotImplementedError(f'target obj {obj} not supported with this HW')

    def _unpack_target(self, target):
        if isinstance(target, tuple):
            assert len(target) == 3
            core, obj, offset = target
        else:
            obj = target
            offset = 0
            core = None
        return core, obj, offset

    def _translate_to_io_msgtype(self, obj : str) -> Tuple[IOTARGET, int]:
        if obj == 'RTR':
            msgtype = 'axis'
            width = 64
        elif obj in SPI_REGS: # SPU system-wide (SPI) registers
            msgtype = 'spu_top'
            width = 32 # actually 16 used bits, but the IO is for 32b words
        elif obj == 'HOST': # commands for the host platform (currently unused)
            msgtype = 'host'
            width = 32
        elif obj in APB_OBJS:
            msgtype = 'apb'
            if obj in CONF_ADDRS:
                width = 64
            else: 
                width = 32 # spu registers
        else:
            assert False # unknown message type

        return msgtype, width

    @classmethod
    def pack_addr_64_to_32(cls, base_addr:int, end_addr:int, length:int) -> ARRAYU32:
        return base_addr, end_addr, length * 2

    @classmethod
    def pack_data_64_to_32(cls, vals:ARRAYU32) -> ARRAYU32:
        datas_msbs = vals >> 32
        datas_lsbs = vals & (2**32 - 1)

        datas_combined = np.zeros((len(vals) * 2,), dtype=np.uint32)

        datas_combined[0::2] = datas_lsbs
        datas_combined[1::2] = datas_msbs

        return datas_combined

    @classmethod
    def pack_64_to_32(cls, base_addr:int, end_addr:int, length:int, vals:ARRAYU64) -> Tuple[ARRAYU32, ARRAYU32]:
        """pack 64b addresses/vals into 32b"""
        return (cls.pack_addr_64_to_32(base_addr, end_addr, length),
                cls.pack_data_64_to_32(vals))

    @classmethod
    def unpack_32_to_64(cls, vals:ARRAYU32) -> ARRAYU64:
        lsbs = vals[0::2]
        msbs = vals[1::2]
        assert(len(lsbs) == len(msbs))

        combined = msbs.astype(np.uint64)
        combined = combined << 32
        combined += lsbs

        return combined

    def hw_send(self,
            target : HWTARGET,
            vals   : Union[ARRAYU64, List[int]],
            flush  : bool=True):
        """send raw data to the hardware

        has different target keys than IOPlugin's hw_send
        allows targetting relative to SPU-specific objects e.g. a particular memory
        IOPlugin flattens this into basic transaction types
        e.g. puts all SPU core controls into APB addr space
        works with 32 and 64b words, which differ by object type
        translates them into 32b words
        """
        vals = np.array(vals, dtype=np.uint64)
        core, obj, offset = self._unpack_target(target)
        addr_range = self._addr_range(obj, offset, len(vals), core=core)
        msgtype, data_width = self._translate_to_io_msgtype(obj)
        if data_width == 64: # break up longer words
            addr_range, vals = SPURunner.pack_64_to_32(*addr_range, vals)

        self.io.hw_send(msgtype, *addr_range, vals, flush)

    def hw_recv(self,
            target    : HWTARGET,
            num_words : int=1) -> List[ARRAYU64]:
        """get raw data back from the hardware

        has different target keys than IOPlugin's hw_send
        allows targetting relative to SPU-specific objects e.g. a particular memory
        IOPlugin flattens this into basic transaction types
        e.g. puts all SPU core controls into APB addr space
        takes 32b words, translates them to
        32 and 64b words, which differ by object type
        """
        core, obj, offset = self._unpack_target(target)
        addr_range = self._addr_range(obj, offset, num_words, core=core)
        msgtype, data_width = self._translate_to_io_msgtype(obj)
        if data_width == 64: # break up longer words
            start_addr, end_addr, length = addr_range
            addr_range = start_addr, end_addr, length * 2
        vals = self.io.hw_recv(msgtype, *addr_range) # 32b words
        if data_width == 64:
            vals = [SPURunner.unpack_32_to_64(v) for v in vals]
        assert(isinstance(vals, list))
        return vals

    def _load_meta_yaml(self, yamlfname:str):
        with open(yamlfname, 'r') as f:
            meta = yaml.safe_load(f)

        # unpack bank sizes
        self.data_bank_sizes = meta['data_bank_sizes']
        self.table_bank_sizes = meta['table_bank_sizes']
        self.inst_counts = meta['inst_counts']

        # unpack mailbox info
        self._mailbox_id_to_precision = {}
        self._mailbox_id_to_varname = {}
        self._simple_output_variable_locs = {}
        for varname, settings in meta['outputs'].items():
            mid = settings['mailbox_id']
            self._mailbox_id_to_precision[mid] = settings['precision']
            self._mailbox_id_to_varname[mid] = varname
            self._simple_output_variable_locs[varname] = mid

    def _hw_write_from_hexfile(self, mem, fname):
        logger.debug(f'LOADING {mem} from file {fname}')
        datas = load_hexfile(fname)
        if len(datas) > 0:
            logger.debug('writing to memory %s', mem)
            if not LIMIT_PROGRAMMING:
                assert False # deprecated, will break burst transactions w/ tables
                self.hw_send(mem, datas)
            else:
                # go bank by bank, only up to used sizes
                cidx, memname, offset = mem
                assert offset == 0
                if memname == 'DM':
                    if cidx in self.data_bank_sizes:
                        for i in range(cfg.CORE_DATA_MEM_BANKS):
                            membank = f'DM{i}'
                            maxidx = self.data_bank_sizes[cidx][i]
                            if maxidx > 0:
                                logger.debug(f'data bank {i}, max size {maxidx}')
                                valid_datas = datas[cfg.DATA_MEM_BANK_WORDS * i : cfg.DATA_MEM_BANK_WORDS * i + maxidx]
                                self.hw_send((cidx, membank, 0), valid_datas)
                elif memname == 'TM':
                    if cidx in self.table_bank_sizes:
                        for i in range(cfg.CORE_TABLE_MEM_BANKS):
                            membank = f'TM{i}'
                            maxidx = self.table_bank_sizes[cidx][i]
                            if maxidx > 0:
                                logger.debug(f'table bank {i}, max size {maxidx}')
                                valid_datas = datas[cfg.TABLE_MEM_BANK_WORDS * i : cfg.TABLE_MEM_BANK_WORDS * i + maxidx]
                                self.hw_send((cidx, membank, 0), valid_datas)
                elif memname == 'IM':
                    maxidx = self.inst_counts[cidx]
                    valid_datas = datas[0 : maxidx]
                    self.hw_send(mem, valid_datas)
                else: 
                    # the others are small
                    assert memname in ['PB', 'RQ', 'SB']
                    self.hw_send(mem, datas)
                    
    
    def get_vars(self, varnames : List[str]) -> VARVALS:
        """read out the values of variables in the hardware"""
        raise NotImplementedError()

    def set_vars(self, set_vals : VARVALS):
        """set the values of variables on the hardware"""
        raise NotImplementedError()

    def soft_reset(self):
        logger.info('soft reset called')
        self.hw_send('RST', [1])
        self.hw_send('RST', [0])

    def change_mem_power_state(self, action:str):
        """controls all memories in used cores"""
        if cfg.ISA == 1.1:
            logger.info(f"tried to change memory action to {action}," +
                    " but nothing to do. Memories always on for TC1")
        elif cfg.ISA == 1.2:
            print(self.data_bank_sizes)
            print(self.table_bank_sizes)
            for core in range(self.used_cores): # used cores
                used_mem_confs = ['IM_CONF']
                if core in self.data_bank_sizes:
                    print(self.data_bank_sizes[core])
                    for i, b in self.data_bank_sizes[core].items():
                        if b > 0:
                            used_mem_confs.append(f'DM_CONF{i}')
                if core in self.table_bank_sizes:
                    for i, b in self.table_bank_sizes[core].items():
                        if b > 0:
                            used_mem_confs.append(f'TM_CONF{i}')
                print(used_mem_confs)
    
                for mem in used_mem_confs: # all core regs are mem ctrls in TC2
                    if action == 'on':
                        logger.info(f'turning on core {core} mem {mem}')
                        self.power_up_mem(core, mem)
                    elif action == 'off':
                        logger.info(f'turning off core {core} mem {mem}')
                        self.power_dn_mem(core, mem)
                    elif action == 'sleep':
                        logger.info(f'sleeping core {core} mem {mem}')
                        self.sleep_mem(core, mem)
                    elif action == 'wake':
                        logger.info(f'waking core {core} mem {mem}')
                        self.wake_mem(core, mem)
        else:
            assert False

    def power_up_mem(self, core:int, mem:str):
        """assumes that the memories are currently off"""
        self.hw_send((core, mem, 0), [MEM_TO_CD])
        self.hw_send((core, mem, 0), [MEM_TO_ON])

    def power_dn_mem(self, core:int, mem:str):
        """assumes that the memories are currently on"""
        self.hw_send((core, mem, 0), [MEM_TO_CD])
        self.hw_send((core, mem, 0), [MEM_TO_OFF])

    def sleep_mem(self, core:int, mem:str):
        """assumes that the memories are currently on"""
        self.hw_send((core, mem, 0), [MEM_TO_CD])
        self.hw_send((core, mem, 0), [MEM_TO_SLEEP_TRANS])
        self.hw_send((core, mem, 0), [MEM_TO_SLEEP])

    def wake_mem(self, core:int, mem:str):
        """assumes that the memories are currently asleep"""
        self.hw_send((core, mem, 0), [MEM_TO_SLEEP_TRANS])
        self.hw_send((core, mem, 0), [MEM_TO_CD])
        self.hw_send((core, mem, 0), [MEM_TO_ON])

    def program_pll(self, multiplier:int):
        """program the PLL, using CLKOD = 1 (min VCO)"""

        assert mutliplier >= 1
        assert mutliplier < 8192

        CLKF = multiplier - 1
        BWADJ = int(multiplier / 2) - 1 # BWADJ is half of CLKF
        CLKOD = 0 # no output division
        CLKR = 0 # no input division

        self.hw_send('PLL_CONF', [1 << 4]) # off

        self.hw_send('PLL_BWADJ', [BWADJ])
        self.hw_send('PLL_BWADJ', [CLKOD])
        self.hw_send('PLL_BWADJ', [CLKF])
        self.hw_send('PLL_BWADJ', [CLKR])

        self.hw_send('PLL_CONF', [0]) # on

    @property
    def used_cores(self):
        return len(self.inst_counts)

    def reset(self, reset_vals=None):
        """reset the program into initial state
        In this case, reprogram the memories, set head/tail
        """

        if reset_vals is not None:
            raise NotImplementedError()
            # I think this is just if you want to reset certain variables?

        self.io.start_apb_recording('0PROG')

        # maybe do some platform-specific reset stuff before programming
        self.io.reset()

        self.change_mem_power_state('on')

        #########################################################3
        # program memories using filled-in ProgState
        basename = self.data_dir + '/' + self.image_fname_pre
        for cidx in range(self.used_cores):
            
            hex_fnames = {}
            hex_fnames['DM'] = basename + f'_core_{cidx}_data_mem_initial_py_hex.txt'
            hex_fnames['TM'] = basename + f'_core_{cidx}_table_mem_initial_py_hex.txt'
            hex_fnames['SB'] = basename + f'_core_{cidx}_sboard_initial_py_hex.txt'
            hex_fnames['RQ'] = basename + f'_core_{cidx}_rqueue_initial_py_hex.txt'
            hex_fnames['PB'] = basename + f'_core_{cidx}_progbuf_initial_py_hex.txt'
            hex_fnames['IM'] = basename + f'_core_{cidx}_instr_mem_initial_py_hex.txt'

            for mem, hex_fname in hex_fnames.items():
                self._hw_write_from_hexfile((cidx, mem, 0), hex_fname)

        self.io.stop_apb_recording()
        self._commit_APB_to_files()

    def finish(self):
        pass

    def reset_hidden_state(self, list_varnames : List[str], sleeptime : float=.030):
        """reset the supplied variables to 0"""
        raise NotImplementedError()
        
    def set_head_tail(self):
        """set the head/tail register to match the state in the program initially
        so any threads that are ready to start (score 0 initially) 
        are effectively placed "on deck"
        """
        # set the head-tail registers in each core
        for core in range(self.used_cores):
            basename = self.data_dir + '/' + self.image_fname_pre
            rqueue_len = np.loadtxt(basename + f'_core_{core}_rqueue_len_initial.txt', dtype=int)
            logger.info('RQUEUE LEN WAS %s', rqueue_len)
            logger.info(f'setting core {core} head-tail reg to have {rqueue_len} threads')
            self.hw_send((core, 'HEAD_TAIL_REG', 0), [rqueue_len])

    def run_all(self, check_exp=False, wait_for_sec=1.0):
        """Run command used by FB unit tests, which have no external IO
        (a thread just starts running--test compares memory states after run to expected values)
        expected values are supplied to the __init__ as mem_expectations
        """
        raise NotImplementedError()

    def _env_SEND_RDY_packer(
            self, 
            send_or_rdy    : str,
            core_idx       : int,
            pc_val         : int,
            packed_payload : Union[None, ARRAYU64]):
        """takes already-packed payload and puts it into router word format:
        prepends the (route, PC) word before data"""

        # this design: one core, or daisy-chained cores, cidx unused

        if send_or_rdy == 'SEND':
            # message stream is just payload with leading PC
            msg_words = np.zeros((packed_payload.shape[0] + 1,), dtype=np.uint64) 
            msg_words[0] = core_idx << 32 | pc_val
            msg_words[1:] = packed_payload
        elif send_or_rdy == 'RDY':
            assert packed_payload is None
            msg_word = core_idx << 32 | pc_val
            msg_words = np.array(msg_word, dtype=np.uint64)
        else:
            raise RuntimeError(f"unknown message type {send_or_rdy}")
        
        return np.atleast_1d(msg_words)

    def _env_RECV_unpacker(self, raw_64b_hw_data_list : List[ARRAYU64]) -> Dict[int, ARRAYU64] :
        """unpacks ((route, mailbox), data, data, ..., data, (route, mailbox), data, data, ..., data), 
        returns dict of {mailbox id : packed payloads}"""
        raise NotImplementedError()

    def env_SEND(self, input_vals : VARVALS):
        """Transmit a SEND from the environment

        converts varnames and packs up element values into 64b words

        Args:
            varvals : dict[str, ARRAYU64] : varnames to element values
        """
        raise NotImplementedError()

    def env_RECV(self) -> VARVALS:
        """Environment performs a RECV
        Assumes that we've waited long enough for everything to come out
        """
        raise NotImplementedError()
        
    def step(self, input_vals : VARVALS) -> VARVALS:
        """Execute one timestep, driving input_vals and getting outputs

        Args:
            input_vals (dict) :
                keyed by variable names, values are numpy arrays for one timestep

        Returns:
            (output_vals, internal_vals) tuple(dict, dict)) :
                tuple of dictionaries with same format as input_vals,
                values for the output variables as well as all internal variables
        """
        raise NotImplementedError()

    def _commit_APB_to_files(self):
        """Dumps captured APB records to files"""
        basedir = 'apb_records'
        os.makedirs(basedir, exist_ok=True)
        for record, datas in self.io.apb_transaction_records.items():
            save_hexfile(basedir + '/' + record + '_A', datas['addr'], bits=32)
            save_hexfile(basedir + '/' + record + '_D', datas['data'], bits=32)

    @classmethod
    def write_APB_files(cls, *runner_args, **runner_kwargs):
        """capture PROG and RHS (reset hidden state) programming files for SD card"""
        hw_runner = cls(*runner_args, fake_connection=True, **runner_kwargs)
        self.reset()
         
class FakeSPURunner(SPURunner):
    def __init__(self, *args, **kwargs):
        """just a SPURunner with fake_connection set true, useful if there's not a convenient way to pass args"""
        super().__init__(*args, fake_connection=True, **kwargs)


