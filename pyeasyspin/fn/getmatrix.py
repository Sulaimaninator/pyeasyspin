"""Read a binary data matrix (Bruker BES3T .DTA etc.)."""
import numpy as np
from .._mat import vec, reshape

_FMT = {'float64': 'f8', 'double': 'f8', 'float32': 'f4', 'single': 'f4', 'float': 'f4', 'int8': 'i1',
        'int16': 'i2', 'int32': 'i4', 'int64': 'i8', 'uint8': 'u1', 'uint16': 'u2', 'uint32': 'u4',
        'uint64': 'u8', 'char': 'i1', 'schar': 'i1', 'uchar': 'u1', 'int': 'i4', 'long': 'i4'}


def getmatrix(FileName=None, Dims=None, NumberFormat=None, ByteOrder=None, isComplex=None):
    """Data array(s) of size Dims; one array per entry of isComplex (a list if several)."""
    bo = '>' if str(ByteOrder).lower() in ('ieee-be', 'b', 'ieee-be.l64', 's') else '<'
    fmt = str(NumberFormat).split('=>')[0].strip().lstrip('*')
    try:
        fh = open(FileName, 'rb')
    except OSError:
        raise ValueError(f'Unable to open data file {FileName}')
    isComplex = vec(isComplex).astype(bool)
    nPer = isComplex.size
    nReals = int(np.sum(isComplex.astype(int) + 1))
    Dims = [int(d) for d in vec(Dims)]
    N = nReals * int(np.prod(Dims))
    with fh:
        data = np.fromfile(fh, dtype=np.dtype(bo + _FMT[fmt]), count=N).astype(float)
    if data.size != N:
        raise ValueError('Unable to read all expected data.')
    rows = list(reshape(data, nReals, int(np.prod(Dims))))
    out = []
    for k in range(nPer):
        if isComplex[k]:
            d = rows[k] + 1j * rows[k + 1]
            del rows[k + 1]
        else:
            d = rows[k]
        D = list(Dims)
        while len(D) > 2 and D[-1] == 1:   # MATLAB drops trailing singleton dimensions
            D.pop()
        out.append(reshape(d, D) if len(D) > 1 else d)
    return out[0] if len(out) == 1 else out
