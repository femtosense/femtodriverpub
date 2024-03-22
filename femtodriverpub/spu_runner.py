try:
    from hardware_config import cfg
except ImportError:
    from femtodriverpub import cfg  # fall back to 1.2 config (eval systems)

from femtorun import FemtoRunner

import femtodriverpub.util.packing as packing
from femtodriverpub.util.hexfile import *

from femtodriverpub.typing_help import VARVALS, ARRAYU64, ARRAYINT, IOTARGET, HWTARGET
from typing import *

from femtodriverpub.plugins.zynq_plugin import ZynqPlugin
from femtodriverpub.plugins.redis_plugin import RedisPlugin


import numpy as np

import os

import yaml

import logging

from femtodriverpub.program_handler import MEM_IMAGE_FNAME_PRE

# this is nothing fancy
# a Mock just allows any method to be called on it, doing nothing
# in this case, NullDebugger will take Debugger's calls and do nothing with them
from unittest.mock import Mock


logger = logging.getLogger(__name__)


class NullDebugger(Mock):
    def get_RHS_vars(self):
        return []

    def get_vars(self, varnames: List[str]) -> VARVALS:
        return {}

    def enter_debug_mode(self) -> None:
        raise RuntimeError("Developer access is needed to enter debug mode.")


# import correct address map for this version
if cfg.ISA == 1.1:
    import femtodriverpub.addr_map_spu1p1 as am
elif cfg.ISA == 1.2:
    import femtodriverpub.addr_map_spu1p2 as am
elif cfg.ISA == 1.3:
    import femtodriverpub.addr_map_spu1p3 as am
elif cfg.ISA == 2.0:
    raise NotImplementedError(
        "ISA 2.0 femtodriverpub not supported. Perhaps you meant to set environment var FS_HW_CFG to something else"
    )
else:
    raise NotImplementedError(f"unrecognized ISA version {cfg.ISA}")

# program/dump only used locations in DM/TM
# saves programming time
LIMIT_PROGRAMMING = True

MEMS = ["DM", "TM", "SB", "RQ", "PB", "IM"]

SEND_TO_RECV_WAIT = 0.1  # seconds

ZYNQ_PLL_MULT = 122  # zynq refclk is about 819KHz, so this is 100M VCO


class SPURunner(FemtoRunner):
    def __init__(
        self,
        data_dir: Union[None, str],
        platform: str = "zcu104",
        debug_vars: Union[str, List[str]] = [],
        mem_expectations: Dict[str, np.ndarray] = {},
        fake_connection: bool = False,
        fake_hw_recv_vals: Union[None, ARRAYU64] = None,
        encrypt: bool = True,
        encryption_key_idx: int = None,
        salt: int = None,
        program_pll: bool = False,
        insert_debug_stores: bool = False,
        compiler_kwargs: Dict = {},
        io_records_dir: str = "io_records",
        **kwargs,
    ):  # catch unused kwargs used by other runners
        """FemtoRunner for real SPU hardware
        Args:
            data_dir : None or str
                directory with memory images and metadata in it
            platform : str (default "zcu104") :
                hardware platform to target (PCB board)
                under the hood, will select a different IOPlugin
            debug_vars : list of strings, or 'all' :
                names of variables to extract values for each run() step()
            mem_expectations : dict[str, np.ndarray] :
                primarily for FB unit test-style run_all(): which variables should have which values at the end of the test
            fake_connection : bool :
                don't actually connect to the hardware, useful if you just want to create an SD card of commands
            encrypt : bool :
                encrypt programming stream
            encryption_key_idx : int (optional)
                index of encryption key to select. if not provided, uses FS_ENCRYPTION_KEY_IDX environment variable or default 0
            salt : int (optional)
                encryption salt. if not provided, uses FS_ENCRYPTION_SALT environment variable or default 0
            program_pll : bool :
                whether or not PLL should be programmed upon reset
            insert_debug_stores : bool :
                manually edit program to insert more stores
            io_records_dir : str :
                where to write IO record yaml
        """
        self.dbg = NullDebugger()

        self.encrypt = encrypt
        self.set_new_data_dir(data_dir)

        self.io_records_dir = io_records_dir

        ##########################################################
        # supported hardware platforms -> IO plugins
        # IO plugin classes must provide hw_send(), hw_recv()
        platform_to_plugin = {
            "zcu104": ZynqPlugin,
            "redis": RedisPlugin,
        }
        self.platform = platform
        self.fake_connection = fake_connection
        if platform in platform_to_plugin:
            self.ioplug = platform_to_plugin[platform](
                fake_connection=fake_connection,
                fake_hw_recv_vals=fake_hw_recv_vals,
                logfiledir=self.io_records_dir,
            )
        else:
            raise NotImplementedError(
                f"unrecognized hardware platform {platform}, must be one of {platform_to_plugin.keys()}"
            )

        # workaround: encrypted disables fast memory programming
        if isinstance(self.ioplug, RedisPlugin):
            self.ioplug.set_encrypted(encrypt)
            logger.info(f"Told redis that encryption is {'on' if encrypt else 'off'}")
            if encrypt:
                logger.warning("Using redis with encryption enabled.")
                logger.warning("Fast memory programming will be disabled.")
                logger.warning("This may take a really long time on large networks.")

        self.should_program_pll = program_pll

        self.debug_vars = debug_vars
        self.expectations = mem_expectations

        # just need to have this key for use w/ FB test
        self.cycles_run = None

        # set encryption details
        if encrypt:
            if encryption_key_idx is None:
                encryption_key_idx = os.getenv("FS_ENCRYPTION_KEY_IDX", default="0")
                try:
                    encryption_key_idx = int(encryption_key_idx)
                except ValueError:
                    logger.error(
                        f"Provided encryption key index {encryption_key_idx} isn't an int"
                    )
                    raise
            logger.info(f"Using encryption key index {encryption_key_idx}")
            self.encryption_key_idx = encryption_key_idx

            if salt is None:
                salt = os.getenv("FS_ENCRYPTION_SALT", default="0")
                try:
                    salt = int(salt, 16)
                except ValueError:
                    logger.error(f"Provided salt {salt} isn't an int")
                    raise
            logger.info(f"Using salt {hex(salt)}")
            self.encryption_salt = salt

    def attach_debugger(self, fasmir):
        try:
            from femtodriverpub.debugger import SPUDebugger

            self.dbg = SPUDebugger(self, fasmir)

            # set up debug vars, expands 'all' debug vars, checks keys
            self.dbg.process_debug_vars()
        except ImportError:
            raise ImportError(
                "Couldn't import debug package. This is a FS-internal feature. Exiting"
            )

    def _addr_range(
        self, obj: str, offset: int, length: int, core: Union[int, None] = None
    ) -> Tuple[int, int, int]:  # base, end, len
        """uses address maps to compute byte addresses from an object string and offset
        note that set_var()/get_var(), used to program data, takes a slightly different route
        still word/element addresses, width depends on target
        """

        if (
            obj in am.CONF_ADDRS
            or obj in am.CORE_REG_REL_ADDRS
            or obj in am.SYS_REG_ADDRS
        ):
            if obj in am.CONF_ADDRS or obj in am.CORE_REG_REL_ADDRS:
                assert core is not None
            elif obj in am.SYS_REG_ADDRS:
                assert core is None

            base_addr = am.OBJ_TO_BYTE_ADDR(obj, core, offset)
            end_addr = am.OBJ_TO_BYTE_ADDR(obj, core, offset + length)
            if obj in am.CONF_ADDRS:
                assert (
                    end_addr - base_addr == 8 * length
                )  # can't cross a bank boundary in one call
            return base_addr, end_addr, length

        elif obj == "RTR":
            assert core is None
            base_addr = end_addr = am.OBJ_TO_BYTE_ADDR(obj)
            return base_addr, end_addr, length

        elif obj in am.SPI_REGS:
            assert core is None
            base_addr = am.OBJ_TO_BYTE_ADDR(obj, core, offset)
            end_addr = am.OBJ_TO_BYTE_ADDR(obj, core, offset + length)
            return base_addr, end_addr, length

        else:
            # something not supported for this HW
            raise NotImplementedError(f"target obj {obj} not supported with this HW")

    def _unpack_target(self, target):
        if isinstance(target, tuple):
            assert len(target) == 3
            core, obj, offset = target
        else:
            obj = target
            offset = 0
            core = None
        return core, obj, offset

    def _translate_to_io_msgtype(self, obj: str) -> Tuple[IOTARGET, int]:
        if obj == "RTR":
            msgtype = "axis"
            width = 64
        elif obj in am.SPI_REGS:  # SPU system-wide (SPI) registers
            msgtype = "spu_top"
            width = 32  # actually 16 used bits, but the IO is for 32b words
        elif obj == "HOST":
            msgtype = "host"
            width = 32
        elif obj in am.APB_OBJS:
            msgtype = "apb"
            if obj in am.CONF_ADDRS:
                width = 64
            else:
                width = 32  # spu registers
        else:
            assert False  # unknown message type

        return msgtype, width

    def hw_send(
        self,
        target: HWTARGET,
        vals: Union[ARRAYU64, List[int]],
        flush: bool = True,
        comment: Optional[str] = None,
    ):
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
        if data_width == 64:  # break up longer words
            addr_range, vals = packing.pack_64_to_32(*addr_range, vals)

        if comment is None:
            comment = f"hw_send to (core, obj, offset) ({core}, {obj}, {offset})"

        self.ioplug.hw_send(msgtype, *addr_range, vals, flush, comment=comment)

    def hw_recv(
        self, target: HWTARGET, num_words: int = 1, comment: Optional[str] = None
    ) -> List[ARRAYU64]:
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
        if data_width == 64:  # break up longer words
            start_addr, end_addr, length = addr_range
            addr_range = start_addr, end_addr, length * 2

        if comment is None:
            comment = f"hw_recv from (core, obj, offset) ({core}, {obj}, {offset})"

        vals = self.ioplug.hw_recv(msgtype, *addr_range, comment=comment)  # 32b words
        if data_width == 64:
            vals = [packing.unpack_32_to_64(v) for v in vals]
        assert isinstance(vals, list)
        return vals

    def set_new_data_dir(self, data_dir: str):
        """update the data dir (meta + images directory)

        note that this includes the call to FR.__init__!

        need to do this in order to be able to update the network on the fly,
        since FR.__init__ takes the padding info
        """

        self.data_dir = data_dir
        self._load_meta_yaml(os.path.join(data_dir, "metadata.yaml"))

        input_padding = None
        output_padding = None
        if "fqir_input_padding" in self.meta:
            input_padding = {
                k: ((v["fqir"],), (v["fasmir"],))
                for k, v in self.meta["fqir_input_padding"].items()
            }
            output_padding = {
                k: ((v["fqir"],), (v["fasmir"],))
                for k, v in self.meta["fqir_output_padding"].items()
            }
        super().__init__(input_padding, output_padding)

        self.input_padding = self.io.input_padding
        self.output_padding = self.io.output_padding

    def _load_meta_yaml(self, yamlfname: str):
        """read in metadata yaml. Do some convenient processing on some entries"""

        with open(yamlfname, "r") as f:
            meta = yaml.safe_load(f)

        self.meta = meta

        # unpack bank sizes
        self.data_bank_sizes = meta["data_bank_sizes"]
        self.table_bank_sizes = meta["table_bank_sizes"]
        self.inst_counts = meta["inst_counts"]

        # unpack mailbox info
        self.output_info = meta["outputs"]
        self.mailbox_id_to_varname = {}
        self.mailbox_id_to_num_words = {}
        for varname, settings in meta["outputs"].items():
            self.mailbox_id_to_varname[settings["mailbox_id"]] = varname
            self.mailbox_id_to_num_words[settings["mailbox_id"]] = settings[
                "len_64b_words"
            ]

        self.input_info = meta["inputs"]

    def _hw_write_from_hexfile(self, mem, fname):
        cidx, memname, _ = mem
        datas = load_hexfile(fname)
        if len(datas) > 0:
            logger.debug("writing to memory %s", mem)
            if not LIMIT_PROGRAMMING:
                assert False  # deprecated, will break burst transactions w/ tables
                self.hw_send(mem, datas)
            else:
                # go bank by bank, only up to used sizes
                cidx, memname, offset = mem
                assert offset == 0
                if memname == "DM":
                    if cidx in self.data_bank_sizes:
                        for i in range(cfg.CORE_DATA_MEM_BANKS):
                            membank = f"DM{i}"
                            maxidx = self.data_bank_sizes[cidx][i]
                            if maxidx > 0:
                                logger.debug(f"data bank {i}, max size {maxidx}")
                                N = cfg.DATA_MEM_BANK_WORDS
                                valid_datas = datas[N * i : N * i + maxidx]
                                self.hw_send((cidx, membank, 0), valid_datas)
                elif memname == "TM":
                    if cidx in self.table_bank_sizes:
                        for i in range(cfg.CORE_TABLE_MEM_BANKS):
                            membank = f"TM{i}"
                            maxidx = self.table_bank_sizes[cidx][i]
                            if maxidx > 0:
                                logger.debug(f"table bank {i}, max size {maxidx}")
                                N = cfg.TABLE_MEM_BANK_WORDS
                                valid_datas = datas[N * i : N * i + maxidx]
                                self.hw_send((cidx, membank, 0), valid_datas)
                elif memname == "IM":
                    maxidx = self.inst_counts[cidx]
                    valid_datas = datas[0:maxidx]
                    self.hw_send(mem, valid_datas)
                else:
                    # the others are small
                    assert memname in ["PB", "RQ", "SB"]
                    self.hw_send(mem, datas)

    def _hw_read_to_hexfile(self, mem, fname, rd_len):
        logger.debug(f"DUMPING {mem} to file {fname}")
        datas = self.hw_recv(mem, num_words=rd_len)[0]
        save_hexfile(fname, datas)

    def get_vars(self, varnames: List[str]) -> VARVALS:
        return self.dbg.get_vars(varnames)

    def set_vars(self, set_vals: VARVALS):
        self.dbg.set_vars(set_vals)

    def soft_reset(self):
        # FIXME enable cores does the same thing
        logger.info("soft reset called")
        self.hw_send("RST", 0b11, comment="soft reset high")
        self.hw_send("RST", 0b00, comment="soft reset low")

    def change_mem_power_state(self, action: str):
        """controls all memories in used cores"""
        # FIXME, should probably only go up to used data banks
        if cfg.ISA == 1.1:
            logger.info(
                f"tried to change memory action to {action},"
                + " but nothing to do. Memories always on for TC1"
            )
        elif cfg.ISA == 1.2 or cfg.ISA == 1.3:
            print(self.data_bank_sizes)
            print(self.table_bank_sizes)
            for core in range(self.used_cores):  # used cores
                used_mem_confs = ["IM_CONF"]
                if core in self.data_bank_sizes:
                    print(self.data_bank_sizes[core])
                    for i, b in self.data_bank_sizes[core].items():
                        if b > 0:
                            used_mem_confs.append(f"DM_CONF{i}")
                if core in self.table_bank_sizes:
                    for i, b in self.table_bank_sizes[core].items():
                        if b > 0:
                            used_mem_confs.append(f"TM_CONF{i}")
                print(used_mem_confs)

                for mem in used_mem_confs:  # all core regs are mem ctrls in TC2
                    if action == "on":
                        logger.info(f"turning on core {core} mem {mem}")
                        self.power_up_mem(core, mem)
                    elif action == "off":
                        logger.info(f"turning off core {core} mem {mem}")
                        self.power_dn_mem(core, mem)
                    elif action == "sleep":
                        logger.info(f"sleeping core {core} mem {mem}")
                        self.sleep_mem(core, mem)
                    elif action == "wake":
                        logger.info(f"waking core {core} mem {mem}")
                        self.wake_mem(core, mem)
        else:
            assert False

    def power_all_memories(self):
        """power on all memories, all cores"""
        for cidx in range(0, cfg.N_CORES):
            for mem_conf_reg_name in am.CORE_REGS:
                self.power_up_mem(cidx, mem_conf_reg_name)

    def powerdown_all_memories(self):
        """power off all memories, all cores"""
        for cidx in range(0, cfg.N_CORES):
            for mem_conf_reg_name in am.CORE_REGS:
                self.power_dn_mem(cidx, mem_conf_reg_name)

    def integrity_check_all_memories(
        self, N: int = 8191
    ) -> Tuple[int, List[List[int]]]:
        """do a single read/write to bottom addr of all memories, all cores
        returns total failures and bit-failures per bank for each core

        args:
        ------
        N : int :
            number of entries tested per memory. Over 2K will have weird effects but might work.
            The smallest memory has 2K entries. I suspect it might just alias.
            8K entries is the max memory size. If N is too large, it will be truncated to the max
            size for each type of memory

        e.g.
            total_fail, by_bank = integrity_check_all_memories(...)
            fails_for_core_idx_bank_idx = by_bank[core_idx][bank_idx]
        bank_idx order: 8 DMs, 4 TMs, 1 IM
        """

        N_max = {
            "DM": 8191,  # for some reason, using 8192 always fails.
            "TM": 2048,
            "IM": 4096,
        }
        all_core_mems = (
            [f"DM{i}" for i in range(8)] + [f"TM{i}" for i in range(4)] + ["IM"]
        )
        bitfail = [[0] * len(all_core_mems) for i in range(cfg.N_CORES)]

        def _test_one_pattern(cidx, test_val_64b: int, test_val_16b: int):
            # compute number of bit-failures per memory

            for mem in all_core_mems:
                self.hw_send((cidx, mem, 0), [test_val_64b] * min(N, N_max[mem[:2]]))

            logger.info("Running memory integrity check")
            for idx, mem in enumerate(all_core_mems):
                read_values = []
                for i in range(min(N, N_max[mem[:2]])):
                    r = self.hw_recv((cidx, mem, i))[0][0]
                    read_values.append(r)

                logger.info(
                    f"first word of {mem} : {int(read_values[0]):016x}, read {len(read_values)} total words"
                )
                if mem.startswith("TM"):
                    test_val = test_val_16b
                else:
                    test_val = test_val_64b

                for read_val in read_values:
                    read_val = int(read_val)
                    # compute number of differing bits per word
                    bitdiff = read_val ^ test_val  # XOR
                    num_ones = bitdiff.bit_count()
                    bitfail[cidx][idx] += num_ones

                if mem == "DM0":
                    numlog = min(N, 100)
                    logger.debug(f"integrity check: first {numlog} DM0 entries")
                    logger.debug(f"  expected: {int(test_val):016x}")
                    for i in range(numlog):
                        logger.debug(f"  {i} : {int(read_values[i]):016x}")

        for cidx in range(0, cfg.N_CORES):
            # test in a checkerboard pattern
            # first write/read 01010101
            # then write/read  10101010
            # this stresses the bitcells by making their neighbors have the opposite bit

            test_val_64b = 0xAAAAAAAAAAAAAAAA
            test_val_16b = 0xAAAA
            _test_one_pattern(cidx, test_val_64b, test_val_16b)

            test_val_64b = 0x5555555555555555
            test_val_16b = 0x5555
            _test_one_pattern(cidx, test_val_64b, test_val_16b)

        total_fails = np.sum(np.array(bitfail))
        fail = total_fails > 0
        bits_tested = cfg.N_CORES * N * (8 * 64 + 4 * 16 + 64)
        # TODO: bits_tested will be wrong for large values of N

        if fail:
            logger.warning(
                f"integrity check of all memories FAILED. fails/bits tested = {total_fails} / {bits_tested}"
            )
        else:
            logger.info("integrity check of all memories was OK")

        return total_fails, bitfail

    def power_up_mem(self, core: int, mem: str):
        """assumes that the memories are currently off"""
        self.hw_send(
            (core, mem, 0), [am.MEM_TO_CD], comment=f"core {core} mem {mem} to CD"
        )
        self.hw_send(
            (core, mem, 0), [am.MEM_TO_ON], comment=f"core {core} mem {mem} to ON"
        )

    def power_dn_mem(self, core: int, mem: str):
        """assumes that the memories are currently on"""
        self.hw_send(
            (core, mem, 0), [am.MEM_TO_CD], comment=f"core {core} mem {mem} to CD"
        )
        self.hw_send(
            (core, mem, 0), [am.MEM_TO_OFF], comment=f"core {core} mem {mem} to OFF"
        )

    def sleep_mem(self, core: int, mem: str):
        """assumes that the memories are currently on"""
        self.hw_send(
            (core, mem, 0), [am.MEM_TO_CD], comment=f"core {core} mem {mem} to CD"
        )
        self.hw_send(
            (core, mem, 0),
            [am.MEM_TO_SLEEP_TRANS],
            comment=f"core {core} mem {mem} to SLEEP_TRANS",
        )
        self.hw_send(
            (core, mem, 0), [am.MEM_TO_SLEEP], comment=f"core {core} mem {mem} to SLEEP"
        )

    def wake_mem(self, core: int, mem: str):
        """assumes that the memories are currently asleep"""
        self.hw_send(
            (core, mem, 0),
            [am.MEM_TO_SLEEP_TRANS],
            comment=f"core {core} mem {mem} to SLEEP_TRANS",
        )
        self.hw_send(
            (core, mem, 0), [am.MEM_TO_CD], comment=f"core {core} mem {mem} to CD"
        )
        self.hw_send(
            (core, mem, 0), [am.MEM_TO_ON], comment=f"core {core} mem {mem} to ON"
        )

    def get_pll_conf_word(
        self, reset=False, bypass=False, pwrdn=False, clkdisable=False
    ):
        # set up conf word
        conf_word = 0
        if reset:
            conf_word |= 1  # reset
        if bypass:
            conf_word |= 1 << 4  # bypass
        if pwrdn:
            conf_word |= 1 << 5  # power down
        if not clkdisable:
            conf_word |= 1 << 6  # post-PLL clock gate is transparent
        return conf_word

    def _program_pll_1p3(
        self, multiplier: int, bypass=False, pwrdn=False, clkdisable=False
    ):
        """program the PLL, using CLKOD = 1 (targeting min VCO)"""

        logger.info("programming PLL")

        assert multiplier >= 1
        assert multiplier < 8192

        CLKF = multiplier - 1
        BWADJ = int(multiplier / 2) - 1  # BWADJ is half of CLKF
        CLKOD = 0  # no output division
        CLKR = 0  # no input division

        # note << 1 for "PLL shift bug" (only for pre-NTO)
        shamt = 0
        if cfg.ISA < 1.3:
            shamt = 1
        self.hw_send("PLL_BWADJ", [BWADJ << shamt])
        self.hw_send("PLL_CLKOD", [CLKOD << shamt])
        self.hw_send("PLL_CLKF", [CLKF << shamt])
        self.hw_send("PLL_CLKR", [CLKR << shamt])

        # power up, reset hi, but keep gate on
        conf = self.get_pll_conf_word(
            reset=True,
            bypass=False,
            pwrdn=False,
            clkdisable=True,
        )
        self.hw_send("PLL_CONF", [conf])

        # reset lo, STILL GATED
        conf = self.get_pll_conf_word(
            reset=False,
            bypass=False,
            pwrdn=False,
            clkdisable=True,
        )
        self.hw_send("PLL_CONF", [conf])

        # wait a little for lock
        logger.info("sleeping .5s to wait for PLL")
        self.wait(200, 0.5)

    def program_pll(self, multiplier: int, bypass=False, pwrdn=False, clkdisable=False):
        if cfg.ISA < 1.3:
            self._program_pll(multiplier, bypass, pwrdn)
        else:
            self._program_pll_1p3(multiplier, bypass, pwrdn, clkdisable)

    def _program_pll(self, multiplier: int, bypass=False, pwrdn=False):
        """program the PLL, using CLKOD = 1 (targeting min VCO)"""

        logger.info("programming PLL")

        assert multiplier >= 1
        assert multiplier < 8192

        CLKF = multiplier - 1
        BWADJ = int(multiplier / 2) - 1  # BWADJ is half of CLKF
        CLKOD = 0  # no output division
        CLKR = 0  # no input division

        # note << 1 for "PLL shift bug" (only for pre-NTO)
        shamt = 0
        if cfg.ISA < 1.3:
            shamt = 1
        self.hw_send("PLL_BWADJ", [BWADJ << shamt])
        self.hw_send("PLL_CLKOD", [CLKOD << shamt])
        self.hw_send("PLL_CLKF", [CLKF << shamt])
        self.hw_send("PLL_CLKR", [CLKR << shamt])

        self.hw_send("PLL_CONF", [1])  # reset hi
        self.hw_send("PLL_CONF", [0])  # reset lo

        # set up conf word
        conf_word = 0
        if bypass:
            conf_word |= 1 << 4  # bypass
        if pwrdn:
            conf_word |= 1 << 5  # power down

        self.hw_send("PLL_CONF", [conf_word])

        # TODO: poll for lock,
        # until then, sleep
        logger.info("sleeping .5s to wait for PLL")
        self.wait(200, 0.5)

    def check_pll_lock(self):
        return self.hw_recv("PLL_LOCK")[0]

    def poll_until_pll_lock(self):
        self.wait(4000, 0.05)
        while self.check_pll_lock()[0] == 0:
            print("polling pll")
            self.wait(4000, 0.05)

    def check_interrupt_reg(self):
        if self.fake_connection:
            print("trying to check interrupt reg for a fake connection")
            print("fake runners should work instantaneously, just returning")
            return np.array([1], dtype=np.uint64)

        if cfg.ISA < 1.3:
            print("trying to check interrupt reg for ISA < 1.3")
            print("sleeping for a second instead")
            print("stuff might break")
            self.wait(1000, 1)
            return np.array([1], dtype=np.uint64)

        return self.hw_recv("OUTPUT_INT")[0]

    def check_version_reg(self):
        return self.hw_recv("VERSION")[0]

    def set_pll_od(self, od: int):
        assert od <= 16
        assert od >= 1

        CLKOD = od - 1
        self.hw_send("PLL_CLKOD", [CLKOD << 1])

        # this sleep is needed for some reason
        self.wait(10, 0.5)

    def write_encryption_regs(self, key_idx, salt):
        if cfg.ISA < 1.3:
            return  # cannot change encryption stuff pre-1.2
        # set encryption key index
        assert key_idx < 32
        self.hw_send("SPI_ENCRYPTION_KEY_IDX", [key_idx])

        # set salt, split across 4 registers
        assert 0 <= salt < 2**64
        for i in range(4):
            salt_part = (salt >> 16 * i) & 0xFFFF
            self.hw_send(f"SPI_ENCRYPTION_SALT_{i}", [salt_part])

    def ungate_pll(self):
        conf = self.get_pll_conf_word(
            reset=False,
            bypass=False,
            pwrdn=False,
            clkdisable=False,
        )
        self.hw_send("PLL_CONF", [conf])

    def enable_cores(self):
        """bring cores out of reset, turn on their clocks"""
        assert cfg.ISA >= 1.3
        self.hw_send(
            (None, "SPI_CONF", 0),
            [1 << 11],
            comment="enable cores, core clocks enabled, reset held",
        )  # enable core clocks but hold core in reset
        self.hw_send(
            (None, "SPI_CONF", 0),
            [0],
            comment="enable cores, core clocks enabled, reset off",
        )  # de-assert core reset
        # interrupt pad
        # curr_io_conf = self.hw_recv((None, 'PAD_CONF', 0))[0]
        # self.hw_send((None, 'PAD_CONF', 0), [curr_io_conf | 1 << 11]) # enable interrupt

    def disable_cores(self):
        """disable core clocks, but don't reset them"""
        self.hw_send((None, "SPI_CONF", 0), [0b110], comment="core clock gates on")

    @property
    def used_cores(self):
        return len(self.inst_counts)

    @property
    def image_basename(self):
        return os.path.join(self.data_dir, MEM_IMAGE_FNAME_PRE)

    def program_network(self):
        """use memory images to program,

        includes all setup intrinsic to the network
        which memory power states

        also sets head/tail"""

        print("POWERING ON MEMORIES")
        self.change_mem_power_state("on")

        # XXX all systems currently enter debug in their FW init, or equivalent thereof
        for cidx in range(self.used_cores):
            hex_fnames = {}
            hex_fnames["DM"] = (
                self.image_basename + f"_core_{cidx}_data_mem_initial_py_hex.txt"
            )
            hex_fnames["TM"] = (
                self.image_basename + f"_core_{cidx}_table_mem_initial_py_hex.txt"
            )
            hex_fnames["SB"] = (
                self.image_basename + f"_core_{cidx}_sboard_initial_py_hex.txt"
            )
            hex_fnames["RQ"] = (
                self.image_basename + f"_core_{cidx}_rqueue_initial_py_hex.txt"
            )
            hex_fnames["PB"] = (
                self.image_basename + f"_core_{cidx}_progbuf_initial_py_hex.txt"
            )
            hex_fnames["IM"] = (
                self.image_basename + f"_core_{cidx}_instr_mem_initial_py_hex.txt"
            )

            for mem, hex_fname in hex_fnames.items():
                print(f"PROGRAMMING {mem}")
                print(f"  fname is {hex_fname}")
                self._hw_write_from_hexfile((cidx, mem, 0), hex_fname)

        # set head/tail register for ready queue
        # this is only needed if there are threads with 0 initial score
        # otherwise, the reset value of 0 is correct
        # FB tests need this, but also need are generally run w/o encryption
        if self.encrypt:
            # the head/tail reg is actually in the encrypted address range,
            # probably need to hardcode some values
            logger.warning("NEED TO IMPLEMENT H/T SETTING WITH ENCRYPTION")
        else:
            self.set_head_tail()

        ## if in debug mode, check what actually got programmed by reading back out from HW
        # if logger.getEffectiveLevel() == logging.DEBUG:
        #    self.dump_mems(basename='debug/dump_initial')

    def init_spu(self):
        """pull hard reset, set up PLL, enabled cores, possibly enter debug mode"""

        # maybe do some platform-specific reset stuff before programming
        self.ioplug.reset()
        self.wait(1000, 3)

        if self.should_program_pll:
            if self.platform == "zcu104":
                self.program_pll(ZYNQ_PLL_MULT)
            if self.platform == "redis":
                # turn on PLL (~100MHz @ .82MHz ref clk) so that memory can be accessed
                # self.program_pll(2000)
                self.program_pll(ZYNQ_PLL_MULT)
            self.poll_until_pll_lock()

        # 1.2 resets to enabled state
        if cfg.ISA >= 1.3:
            # open up the master gate, core gates already open
            print("ENABLING CORES")
            self.ungate_pll()
            self.enable_cores()

        if not self.encrypt:
            self.dbg.enter_debug_mode()
        else:
            self.write_encryption_regs(self.encryption_key_idx, self.encryption_salt)

    def reset_hidden_state(self):
        # if init'd from FQIR, we know which variables are in the init part of the graph
        # can write "reset hidden state" file (RHS) for these
        # note that init is always to 0 right now--might not always work!
        if self.encrypt:
            logger.warning("NEED TO IMPLEMENT RHS WITH ENCRYPTION")
        else:
            if self.dbg.fqir is not None:
                # XXX FIXME, might need to pass some other kind of metadata for this
                # or extend FASMIR in general for more general reset sequence
                self._reset_hidden_state()
            else:
                logger.warning(
                    "NO DEBUGGER WITH FQIR, NOT COLLECTING RESET HIDDEN STATE (RHS)"
                )

    def reset(self, reset_vals=None, record=True):
        """reset the program into initial state
        In this case, reprogram the memories, set head/tail
        """
        if reset_vals is not None:
            raise NotImplementedError()
            # I think this is just if you want to reset certain variables?

        if record:
            self.ioplug.start_recording("0PROG")

        # set up clocks, power memories
        self.init_spu()

        # program memories
        self.program_network()

        # reset hidden states (not strictly necessary,
        # we just do this to get the record file
        if record:
            self.ioplug.start_recording("RHS")
            self.reset_hidden_state()

        if record:
            self.ioplug.commit_recording("setup.yaml")
            self.commit_APB_to_files()

    def finish(self):
        # self.ioplug.commit_recording("all.yaml")
        pass

    def _reset_hidden_state(self, sleeptime: float = 0.030):
        """reset the supplied variables to 0"""
        # XXX extend to non-zero RHS vals
        reset_vars = self.dbg.get_RHS_vars()
        var_objs = self.dbg.get_vars(reset_vars)
        reset_varvals = {
            var: np.zeros(var_objs[var].data.numpy.shape, dtype=np.int64)
            for var in reset_vars
        }
        self.set_vars(reset_varvals)
        self.wait(50, sleeptime)

    def set_head_tail(self):
        """set the head/tail register to match the state in the program initially
        so any threads that are ready to start (score 0 initially)
        are effectively placed "on deck"
        """

        # set the head-tail registers in each core
        for core in range(self.used_cores):
            rqueue_len = np.loadtxt(
                self.image_basename + f"_core_{core}_rqueue_len_initial.txt", dtype=int
            )
            logger.info("RQUEUE LEN WAS %s", rqueue_len)
            logger.info(
                f"setting core {core} head-tail reg to have {rqueue_len} threads"
            )
            self.hw_send((core, "HEAD_TAIL_REG", 0), [rqueue_len])

    def run_all(self, check_exp=False, wait_for_sec=1.0):
        """Run command used by FB unit tests, which have no external IO
        (a thread just starts running--test compares memory states after run to expected values)
        expected values are supplied to the __init__ as mem_expectations
        """

        self.reset()  # program, set H/T (which should start some thread(s) automatically)

        logger.info("waiting")
        self.wait(1000, wait_for_sec)

        # if logger.getEffectiveLevel() == logging.DEBUG:
        #    self.dump_mems(basename='debug/dump_final')

        for var in self.debug_vars:
            logger.debug("getting value of debug_var %s", var)
            logger.debug("%s", str(self.get_vars([var])))

        logger.info("run over, reading expected vars")
        extracted_vals = self.get_vars(self.expectations.keys())
        if (
            FemtoRunner.compare_outputs(
                "expected", self.expectations, "hardware", extracted_vals
            )
            != 0
        ):
            raise ValueError(
                "OUTPUT COMPARISONS FAILED! See log"
            )  # match format pytest expects

        return extracted_vals

    def dump_mems(self, basename=None):
        # dump memory contents
        if not LIMIT_PROGRAMMING:
            mem_sizes = {
                "DM": cfg.CORE_DATA_MEM_WORDS,
                "TM": cfg.CORE_TABLE_MEM_WORDS,
                "SB": cfg.MAX_THREADS,
                "RQ": cfg.MAX_THREADS,
                "PB": cfg.MAX_THREADS,
                "IM": cfg.MAX_INSTR,
            }
        else:
            # used as bank sizes
            mem_sizes = {
                "DM": 64,
                "TM": 64,
                "SB": cfg.MAX_THREADS,
                "RQ": cfg.MAX_THREADS,
                "PB": cfg.MAX_THREADS,
                "IM": 32,
            }

        if basename is None:
            basename = os.path.join(self.data_dir, "dump")

        for cidx in range(self.used_cores):
            # this is just to get filenames, we'll overwrite these
            filenames = self.prog_state[cidx].save_packed_mems(
                basename, "", use_core_str=True
            )

            for mem in MEMS:
                hex_fnames = filenames["hex"][mem]  # could contain multiple bitgroups
                assert len(hex_fnames) == 1  # but assert there's just one
                hex_fname = hex_fnames[0]

                if not LIMIT_PROGRAMMING:
                    self._hw_read_to_hexfile((cidx, mem, 0), hex_fname, mem_sizes[mem])
                else:
                    if mem == "DM":
                        banks = cfg.CORE_DATA_MEM_BANKS

                        def offsetfn(i):
                            return i * cfg.DATA_MEM_BANK_WORDS

                    elif mem == "TM":
                        banks = cfg.CORE_TABLE_MEM_BANKS

                        def offsetfn(i):
                            return i * cfg.TABLE_MEM_BANK_WORDS

                    else:
                        banks = 1

                    for i in range(banks):
                        self._hw_read_to_hexfile(
                            (cidx, mem, offsetfn(i)),
                            hex_fname + f"_b{i}",
                            mem_sizes[mem],
                        )

    def _env_SEND_RDY_packer(
        self,
        send_or_rdy: str,
        core_idx: int,
        pc_val: int,
        packed_payload: Union[None, ARRAYU64],
    ):
        """takes already-packed payload and puts it into router word format:
        prepends the (route, PC) word before data"""

        # this design: one core, or daisy-chained cores, cidx unused

        if send_or_rdy == "SEND":
            # message stream is just payload with leading PC
            msg_words = np.zeros((packed_payload.shape[0] + 1,), dtype=np.uint64)
            msg_words[0] = pc_val << 32 | core_idx
            msg_words[1:] = packed_payload
        elif send_or_rdy == "RDY":
            assert packed_payload is None
            msg_word = pc_val << 32 | core_idx
            msg_words = np.array(msg_word, dtype=np.uint64)
        else:
            raise RuntimeError(f"unknown message type {send_or_rdy}")

        return np.atleast_1d(msg_words)

    def _env_RECV_unpacker(
        self, raw_64b_hw_data_list: List[ARRAYU64]
    ) -> Dict[int, ARRAYU64]:
        """unpacks ((route, mailbox), data, data, ..., data, (route, mailbox), data, data, ..., data),
        returns dict of {mailbox id : packed payloads}"""

        mailbox_to_packed = {}
        next_word_is_header = True
        words_to_read = 0
        curr_payload = []
        for msg in raw_64b_hw_data_list:
            for word in msg:
                if next_word_is_header:
                    mailbox = int(word) >> 32
                    words_to_read = self.mailbox_id_to_num_words[mailbox]
                else:
                    words_to_read -= 1
                    assert words_to_read >= 0
                    curr_payload.append(word)

                if words_to_read == 0:
                    next_word_is_header = True
                    mailbox_to_packed[mailbox] = np.array(curr_payload, dtype=np.uint64)
                else:
                    next_word_is_header = False

        return mailbox_to_packed

    def env_SEND(self, input_vals: VARVALS):
        """Transmit a SEND from the environment

        converts varnames and packs up element values into 64b words

        Args:
            varvals : dict[str, ARRAYU64] : varnames to element values
        """
        for varname, element_vals in input_vals.items():
            # look up varname -> core/pc/precision
            settings = self.input_info[varname]

            # pack it up
            payload = packing.pack_V(
                settings["precision"], element_vals
            )  # element vals -> 64b words
            assert settings["len_64b_words"] == len(payload)
            assert settings["dtype"][0] == "V"
            msg = self._env_SEND_RDY_packer(
                "SEND", settings["core"], settings["pc_val"], payload
            )

            # send it
            logger.debug(
                "about to transmit env_SEND() for var %s (core %d pc_val %s)",
                varname,
                settings["core"],
                settings["pc_val"],
            )
            self.hw_send(
                "RTR", msg, comment=f"env_SEND to vars {list(input_vals.keys())}"
            )

    def env_RECV(self, total_words=None) -> VARVALS:
        """Environment performs a RECV
        Assumes that we've waited long enough for everything to come out
        """
        # receive all output vars, looking at fasmir to know how big they are
        if total_words is None:
            total_words = 0
            for output_var, settings in self.output_info.items():
                total_words += settings["len_64b_words"]
                # XXX do we need this? maybe there's some funny business deeper in?
                # total_words += 1 # for route word
                # if we don't need this, we are likely to break for > 1 output, because there's something hardcoded deeper in

        logger.debug(f"env_RECV : about to try to grab {total_words} words")

        # ask for data
        messages = self.hw_recv("RTR", num_words=total_words, comment="env_RECV")

        # convert mailbox ids to varnames, unpack data
        mailbox_to_packed_data = self._env_RECV_unpacker(messages)

        element_vals = {}
        for mid, data in mailbox_to_packed_data.items():
            varname = self.mailbox_id_to_varname[mid]
            settings = self.output_info[varname]
            assert settings["len_64b_words"] == len(data)
            assert settings["dtype"][0] == "V"
            element_vals[varname] = packing.unpack_V(settings["precision"], data)

        for varname, vals in element_vals.items():
            logger.debug(f"env_RECV : element values for {varname}")
            logger.debug(f"  {vals}")

        return element_vals

    def wait(self, sim_cycles, real_seconds):
        if isinstance(self.ioplug, RedisPlugin):
            self.ioplug.wait(sim_cycles)
        elif isinstance(self.ioplug, ZynqPlugin):
            self.ioplug.wait(real_seconds)

    def step(self, input_vals: VARVALS) -> VARVALS:
        """Execute one timestep, driving input_vals and getting outputs

        Args:
            input_vals (dict) :
                keyed by variable names, values are numpy arrays for one timestep

        Returns:
            (output_vals, internal_vals) tuple(dict, dict)) :
                tuple of dictionaries with same format as input_vals,
                values for the output variables as well as all internal variables
        """
        # apply padding if nominal input length differs from implemented padded size
        # this is implemented in FR base
        padded_inputs = self._pad_inputs(input_vals)

        # send inputs in
        self.env_SEND(padded_inputs)

        # sleep for a bit, just to make sure processing is done
        self.wait(1000, SEND_TO_RECV_WAIT)

        # now poll for interrupt
        while self.check_interrupt_reg()[0] == 0:
            print("polling for interrupt")
            # SPI clock is typically run at 100MHz in simuation for faster overall speed
            # this is unrealistically fast, but .05ms @ 100MHz is 5K cycles
            # this is a lot of cycles--we can affort to check more often
            self.wait(1000, 0.05e-3)

        # get internal values
        # do this before trying to get output, this is probably a bit less error-prone
        internal_vals = self.get_vars(self.debug_vars)

        # return the outputs
        padded_outputs = self.env_RECV()

        # undo padding on the way out
        # this is implemented in FR base
        output_vals = self._unpad_outputs(padded_outputs)

        return output_vals, internal_vals

    def make_fake_inputs(self, T, style="sawtooth"):
        """generate a set of T input to send to each input variable"""
        # collect the input variables
        fake_inputs = {}
        for varname, settings in self.input_info.items():
            if settings["precision"] == "STD":
                D = settings["len_64b_words"] * 8
            else:
                D = settings["len_64b_words"] * 4

            if style == "sawtooth":
                fake_inputs[varname] = (np.arange(T * D) % 32).reshape((T, D))
            elif style == "random":
                fake_inputs[varname] = (
                    np.random.randint(2**15, size=(T * D)) - 2**14
                ).reshape(
                    (T, D)
                ) // 8  # don't do max-size
        return fake_inputs

    def get_single_output_name(self):
        """assert the model has one output, return its name"""
        assert len(self.meta["outputs"]) == 1
        return next(iter(self.meta["outputs"]))

    def run(self, input_val_timeseries: Dict[str, ARRAYINT]):
        """Execute several timesteps, iterating through the values of input_val_timeseries
        driving the runner at each timestep. Calls .step() each timestep.

        Here we modify Femtorunner's run() to add an memory dump if we're in DEBUG

        Args:
            input_val_timeseries (dict {varname (str): value (numpy.ndarray)}):
                keyed by variable names, values are numpy arrays, first dim is time

        Returns:
            (output_vals, internal_vals, output_valid_mask) tuple(dict, dict, dict)):
                tuple of dictionaries with same format as input_vals,
                values for the output variables as well as all internal variables,
                for all timesteps that were run
                output_valid_mask contains a 1D bool array for each key, says whether each output
                was produced on a given timestep
        """
        to_ret = super().run(input_val_timeseries)

        # if logger.getEffectiveLevel() == logging.DEBUG:
        #    self.dump_mems(basename='debug/dump_final')

        return to_ret

    def commit_APB_to_files(self):
        """Dumps captured APB records to files"""
        PROG_path = os.path.join(self.io_records_dir, "apb_records")
        if not os.path.exists(PROG_path):
            os.makedirs(PROG_path)
        self.ioplug.records.dump_apb_to_text(PROG_path)

    @classmethod
    def write_APB_files(cls, *runner_args, **runner_kwargs):
        """capture PROG and RHS (reset hidden state) programming files for SD card"""
        hw_runner = cls(*runner_args, fake_connection=True, **runner_kwargs)
        hw_runner.reset()


class FakeSPURunner(SPURunner):
    def __init__(self, *args, **kwargs):
        """just a SPURunner with fake_connection set true, useful if there's not a convenient way to pass args"""
        super().__init__(*args, fake_connection=True, **kwargs)
