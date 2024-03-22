# typing convenience
from typing import Dict, Tuple, Union, Literal
import numpy as np
import numpy.typing as npt

ARRAYINT = npt.NDArray[int]
ARRAYU64 = npt.NDArray[np.uint64]
ARRAYU32 = npt.NDArray[np.uint32]
VARVALS = Dict[str, ARRAYINT]
VARPACKED = Dict[str, ARRAYU64]

"""
HWTARGET is one of:
something inside an SPU core:
    tuple(int: core, str: hw object, offset) 
something there's only one of on the chip:
    str : hw object (offset=0)
"""
HWTARGET = Union[Tuple[int, str, int], str]

"""
what the IOPlugin uses for it's hw_send
each of these could be handled differently on different platforms
RTR           -> axis
SPI_REGS      -> spu_top
HOST          -> host
anything else -> apb
"""
IOTARGET = Literal["apb", "axis", "host", "spu_top"]
