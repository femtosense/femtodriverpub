import numpy as np


def load_hexfile(filename):
    try:
        vals = np.loadtxt(
            filename, dtype="uint64", usecols=(0,), converters={0: lambda s: int(s, 16)}
        )
    except UserWarning:
        vals = np.array([], dtype=np.uint64)
    return np.atleast_1d(vals)


def save_hexfile(fname, vals, bits=None):
    def bits_to_fmt(x):
        return "%0" + str(int(np.ceil(x) / 4)) + "x"

    def hex_fmt_for_mem(mem):
        membits = {
            "DM": cfg.B_DATA_WORD,
            "TM": cfg.B_TABLE_WORD,
            "SB": cfg.B_SB,
            "RQ": cfg.B_RQ,
            "PB": cfg.B_PC,
            "IM": cfg.B_INSTR,
        }
        return bits_to_fmt(membits[mem])

    # np.savetxt(fname, vals, fmt=hex_fmt_for_mem(mem))
    # XXX should infer memory type
    if bits is None:
        fmt = hex_fmt_for_mem("DM")
    else:
        fmt = bits_to_fmt(bits)
    np.savetxt(fname, np.atleast_1d(vals), fmt=fmt)
