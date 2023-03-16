from femtodriverpub import cfg

APB_ADDR         = 0x0 # base of APB addr range

B_CONF_ADDR = cfg.B_DATA_ADDR + 1
CONF_BLOCK = 2**B_CONF_ADDR
BYTE_CONF_BLOCK = 2**B_CONF_ADDR * 8
BANK_BLOCK = cfg.DATA_MEM_BANK_WORDS

#############################################
# 16b SPI reg space
# still programmable with 32b words
SPI_REGS = [
    'SPI_CONF',  # basic conf settings (core clock gating)
    'PAD_CONF',  # pad IO settings, including osc pad
    'PLL_CONF',  # PLL control bits
    'PLL_BWADJ', # bandwidth adjust, half of CLKF
    'PLL_CLKOD', # output divider
    'PLL_CLKF',  # frequency multiplier
    'PLL_CLKR',  # pre-VCO divider
]
SPI_REG_TO_IDXS = {reg : i for i, reg in enumerate(SPI_REGS)}

N_CORES = 2
##############################################
# 32b control register space
# register space has SYS_REGS below one block of CORE_REGs per core
# but packed at the interval of CORE_REGS
#
# so:
# 
# [0 - 4]  : sys regs
# [5 - 10] : empty
# [11 : 21] : core 0 regs
# [22 : 32] : core 1 regs
# etc.

SYS_REGS = [
    'VERSION',           # HW version, 8b, {4b: major , 4b: minor}
    'RST',               # core reset
    'DM_TIMERS',         # automatic sleep FSM regs
    'TM_TIMERS',
    'IM_TIMERS',         
    'REG_DP_ACK_DELAY',  # how many fudge cycles to pad on the end of the datapaths' ack
]

# useful mem power state values:
# (default timing adjust pin values)
# to "off"                    : 0x01098
# to "chip disable (CD)" hub  : 0x05098
# to "sleep"                  : 0x09098
# to "sleep trans"            : 0x0d098
# to "on"                     : 0x11098
# to "FSM control"            : 0x15098
#
# allowed transitions
# off <-> CD
# on <->
# sleep_trans <-> CD
# sleep_trans <-> sleep
#
# e.g.:
# power on: 
#   off -> CD -> on
# power off:
#   on  -> CD -> off
# powered -> retention:
#   on  -> CD -> sleep_trans -> sleep
# retention -> powered:
#   sleep -> sleep_trans -> CD -> on
CORE_REGS = [
    *[f'DM_CONF{i}' for i in range(8)], # power state toggles, delay adjust values (EMA, eg)
    *[f'TM_CONF{i}' for i in range(4)], 
    'IM_CONF',
]

SYS_REG_ADDRS = {r : i for i, r in enumerate(SYS_REGS)}
CORE_REG_REL_ADDRS = {r : i for i, r in enumerate(CORE_REGS)}

CONF_ADDRS = {
        'DM'            : 0                                                                         ,
        'TM'            : 0 + BANK_BLOCK * (cfg.CORE_DATA_MEM_BANKS + 0)                            ,
        'SB'            : 0 + BANK_BLOCK * (cfg.CORE_DATA_MEM_BANKS + cfg.CORE_TABLE_MEM_BANKS + 0) ,
        'RQ'            : 0 + BANK_BLOCK * (cfg.CORE_DATA_MEM_BANKS + cfg.CORE_TABLE_MEM_BANKS + 1) ,
        'PB'            : 0 + BANK_BLOCK * (cfg.CORE_DATA_MEM_BANKS + cfg.CORE_TABLE_MEM_BANKS + 2) ,
        'IM'            : 0 + BANK_BLOCK * (cfg.CORE_DATA_MEM_BANKS + cfg.CORE_TABLE_MEM_BANKS + 3) ,
        'HEAD_TAIL_REG' : 0 + CONF_BLOCK-3                                                          ,
}

# system register useful const values:
MEM_TO_OFF         = 0x01098
MEM_TO_CD          = 0x05098
MEM_TO_SLEEP       = 0x09098
MEM_TO_SLEEP_TRANS = 0x0d098
MEM_TO_ON          = 0x11098
MEM_TO_FSM         = 0x15098

# add addresses for individual banks, allow us to program one bank at a time
for i in range(cfg.CORE_DATA_MEM_BANKS):
    CONF_ADDRS[f'DM{i}'] = CONF_ADDRS['DM'] + BANK_BLOCK * i
for i in range(cfg.CORE_TABLE_MEM_BANKS):
    CONF_ADDRS[f'TM{i}'] = CONF_ADDRS['TM'] + BANK_BLOCK * i

APB_OBJS = set(SYS_REGS) | set(CORE_REGS) | set(CONF_ADDRS.keys())

# NOTE THAT HEAD_TAIL_REG HAS MOVED FROM ITS TRADITIONAL CONF_BLOCK-1 ADDR!
# now the MEM_POWERs go in -1 and -2

MEMNAMES = {
    'DM',
    'TM',
    'SB',
    'RQ',
    'PB',
    'IM',
}

def mem_conf_addr_map(mem, file_offset):
    base_addr = CONF_ADDRS[mem]

    if mem == 'TM':
        addr_offset = BANK_BLOCK * (file_offset // cfg.TABLE_MEM_BANK_WORDS) + file_offset % cfg.TABLE_MEM_BANK_WORDS
    else:
        addr_offset = file_offset

    addr = base_addr + addr_offset
    #print(f'mem: {mem}, file_offset {file_offset}, addr {addr}')

    return addr

def CONF_ADDR_TO_BYTE_ADDR(core, rel_conf_addr):
    return APB_ADDR + (core + 1) * BYTE_CONF_BLOCK + rel_conf_addr * 8
def SYS_REG_ADDR_TO_BYTE_ADDR(core, rel_reg_addr): # core is unused
    return APB_ADDR + rel_reg_addr * 4
def CORE_REG_ADDR_TO_BYTE_ADDR(core, rel_reg_addr):
    return APB_ADDR + (core + 1) * max(len(SYS_REGS), len(CORE_REGS)) * 4 + rel_reg_addr * 4

def OBJ_TO_BYTE_ADDR(obj, core=None, file_offset=0):
    if obj in MEMNAMES or len(obj) == 3 and (obj.startswith('DM') or obj.startswith('TM')):
        conf_addr = mem_conf_addr_map(obj, file_offset)
        axi_addr = CONF_ADDR_TO_BYTE_ADDR(core, conf_addr)
    elif obj in CONF_ADDRS:
        conf_addr = CONF_ADDRS[obj]
        axi_addr = CONF_ADDR_TO_BYTE_ADDR(core, conf_addr + file_offset)
    elif obj in SYS_REGS:
        reg_addr = SYS_REG_ADDRS[obj]
        axi_addr = SYS_REG_ADDR_TO_BYTE_ADDR(core, reg_addr + file_offset)
    elif obj in CORE_REGS:
        reg_addr = CORE_REG_REL_ADDRS[obj]
        axi_addr = CORE_REG_ADDR_TO_BYTE_ADDR(core, reg_addr + file_offset)
    elif obj == 'RTR':
        assert file_offset == 0
        axi_addr = 0 # filled in by zynq
    elif obj in SPI_REGS:
        return (SPI_REG_TO_IDXS[obj] + file_offset) * 4
    else:
        axi_addr = None
    return axi_addr

