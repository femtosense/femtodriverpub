import math
from femtodriverpub.typing_help import ARRAYU64, ARRAYU32, ARRAYINT
from typing import Tuple, Literal
import numpy as np


def maxval_signed(B):
    return (1 << (B - 1)) - 1


def minval_signed(B):
    return -(1 << (B - 1))


def maxval_unsigned(B):
    return (1 << B) - 1


def minval_unsigned(B):
    return 0


def minval(signedness, B):
    if signedness == "s":
        return minval_signed(B)
    elif signedness == "u":
        return minval_unsigned(B)
    else:
        assert False


def maxval(signedness, B):
    if signedness == "s":
        return maxval_signed(B)
    elif signedness == "u":
        return maxval_unsigned(B)
    else:
        assert False


def in_range_signed(x, B):
    """Checks that all values in x are representable with B-bits, signed"""
    return np.max(x) <= maxval_signed(B) and np.min(x) >= minval_signed(B)


def in_range_unsigned(x, B):
    """Checks that all values in x are representable with B-bits, unsigned"""
    return np.max(x) <= maxval_unsigned(B) and np.min(x) >= 0


def assert_in_range_signed(x, B):
    """Checks that all values in x are representable with B-bits, signed"""
    if not in_range_signed(x, B):
        raise ValueError("trying to set signed {}'b{}".format(B, x))


def assert_in_range_unsigned(x, B):
    """Checks that all values in x are representable with B-bits, unsigned"""
    if not in_range_unsigned(x, B):
        raise ValueError("trying to set unsigned {}'b{}".format(B, x))


def unsigned_to_signed(x_unsigned, B):
    assert_in_range_unsigned(x_unsigned, B)
    max_pos = maxval_signed(B)
    if isinstance(x_unsigned, np.ndarray):
        x_signed = x_unsigned.copy()
        x_signed[x_unsigned > max_pos] -= 2**B
        return x_signed.astype(np.int64)
    else:
        if x_unsigned > max_pos:
            return x_unsigned - 2**B
        else:
            return x_unsigned


def signed_to_unsigned(x_signed, B):
    """Convert signed int value to bit-equivalent unsigned value

    e.g. in 8-bits
        signed -> unsigned
           127 -> 127
              ...
             1 -> 1
             0 -> 0
            -1 -> 255
              ...
          -128 -> 128
    """
    assert_in_range_signed(x_signed, B)
    if isinstance(x_signed, np.ndarray):
        x_unsigned = x_signed.copy()
        x_unsigned[x_unsigned < 0] += 2**B
        return x_unsigned
    else:
        if x_signed < 0:
            return x_signed + 2**B
        else:
            return x_signed


def prec_elts_per_word(precision: str) -> int:
    assert 64 % prec_bits(precision) == 0
    return int(64 / prec_bits(precision))


def prec_bits(precision: str) -> int:
    if precision == "STD":
        return 8
    elif precision == "DBL":
        return 16
    else:
        assert False


def words_to_els(words: int, precision: str) -> int:
    return words * prec_elts_per_word(precision)


def els_to_words(els: int, precision: str, assert_even: bool = False) -> int:
    n_per_word = prec_elts_per_word(precision)
    if assert_even:
        assert els % n_per_word == 0
    return int(math.ceil(els / n_per_word))


def pack_V(precision: str, el_vals: ARRAYINT) -> ARRAYU64:
    """convert signed data element array (e.g. one 16b signed per entry) to packed uint64"""
    B = prec_bits(precision)
    num_words = els_to_words(len(el_vals), precision)

    elts_per_word = prec_elts_per_word(precision)

    packed = np.full((num_words,), 0, dtype=np.uint64)

    for i in range(num_words):
        for j in range(elts_per_word):
            signed_el = el_vals[i * elts_per_word + j]
            unsigned_el = signed_to_unsigned(signed_el, B)
            packed[i] |= np.uint64(unsigned_el << (B * j))

    return packed


def unpack_V(precision: str, packed_vals: ARRAYU64) -> ARRAYINT:
    """convert packed uint64 to signed data element array (e.g. one 16b signed per entry)"""
    B = prec_bits(precision)
    num_words = len(packed_vals)
    num_elts = words_to_els(num_words, precision)

    elts_per_word = prec_elts_per_word(precision)
    unpacked = np.full((num_elts,), 0, dtype=np.int64)

    for i in range(num_words):
        for j in range(elts_per_word):
            mask = np.uint64(((1 << B) - 1) << (B * j))
            unsigned_el = (mask & np.uint64(packed_vals[i])) >> np.uint64(B * j)
            unpacked[i * elts_per_word + j] = unsigned_to_signed(unsigned_el, B)

    return unpacked


def pack_addr_64_to_32(base_addr: int, end_addr: int, length: int) -> ARRAYU32:
    return base_addr, end_addr, length * 2


def pack_data_64_to_32(vals: ARRAYU32) -> ARRAYU32:
    datas_msbs = vals >> 32
    datas_lsbs = vals & (2**32 - 1)

    datas_combined = np.zeros((len(vals) * 2,), dtype=np.uint32)

    datas_combined[0::2] = datas_lsbs
    datas_combined[1::2] = datas_msbs

    return datas_combined


def pack_64_to_32(
    base_addr: int, end_addr: int, length: int, vals: ARRAYU64
) -> Tuple[ARRAYU32, ARRAYU32]:
    """pack 64b addresses/vals into 32b"""
    return (pack_addr_64_to_32(base_addr, end_addr, length), pack_data_64_to_32(vals))


def unpack_32_to_64(vals: ARRAYU32) -> ARRAYU64:
    lsbs = vals[0::2]
    msbs = vals[1::2]
    assert len(lsbs) == len(msbs)

    combined = msbs.astype(np.uint64)
    combined = combined << 32
    combined += lsbs

    return combined


def pack_like_reply(precision: Literal["STD", "DBL"], vals: np.ndarray):
    lvals = list(vals)
    if precision == "STD":
        with_header = [0] * 8 + lvals  # values should be disregarded
    elif precision == "DBL":
        with_header = [0] * 4 + lvals  # values should be disregarded
    else:
        assert False
    array_again = np.array(with_header, dtype=int)
    return pack_V(precision, array_again)
