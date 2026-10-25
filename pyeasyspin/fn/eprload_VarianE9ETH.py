"""Varian E9 ETH (.spk/.ref) loader."""
import numpy as np
from .._registry import nout
from .._io import BinFile


@nout(3)
def eprload_VarianE9ETH(FileName=None):
    """(Data row, Abscissa=[], Parameters=[]): the last 500/1000/2000/5000/10000 float32 values."""
    try:
        RawData = BinFile(FileName, 'ieee-le').read(None, 'float32')
    except OSError:
        raise ValueError(f'Could not open {FileName}.')
    N = RawData.size
    K = [k for k in (500, 1000, 2000, 5000, 10000) if N > k]
    if not K:
        raise ValueError('File too small.')
    Data = RawData[N - K[-1]:].reshape(1, -1)
    return Data, np.zeros((0, 0)), np.zeros((0, 0))
