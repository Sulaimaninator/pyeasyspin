"""ETH/WIS (.d00) loader."""
import numpy as np
from .._registry import nout
from .._mat import reshape
from .._io import BinFile


@nout(3)
def eprload_d00WISETH(FileName=None):
    """(complex Data of the stored dimensions, Abscissa=[], Parameters=[])."""
    try:
        h = BinFile(FileName, 'ieee-le')
    except OSError:
        raise ValueError('Could not open ' + FileName)
    Dims = [int(d) for d in h.read(3, 'int16')]
    raw = h.read(None, 'double')
    raw = raw[:raw.size // 2 * 2]
    Data = raw[0::2] + 1j * raw[1::2]
    while len(Dims) > 2 and Dims[-1] == 1:
        Dims.pop()
    return reshape(Data, Dims), np.zeros((0, 0)), np.zeros((0, 0))
