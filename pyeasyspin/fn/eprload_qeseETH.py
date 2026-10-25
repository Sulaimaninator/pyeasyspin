"""qese/tryscore (.eco) loader."""
import re
import numpy as np
from .._registry import nout
from .._mat import reshape
from .._io import fgetl_lines, sscanf_f


@nout(3)
def eprload_qeseETH(FileName=None):
    """(Data, Abscissa=[], Parameters=[]) from an ASCII .eco file (header: dims [isComplex])."""
    try:
        lines = fgetl_lines(FileName)
    except OSError:
        raise ValueError('Could not open ' + FileName)
    head = []
    for m in re.finditer(r'\S+', lines[0]):
        try:
            head.append(int(m.group(0), 0))
        except ValueError:
            break
        if len(head) == 3:
            break
    if len(head) == 3:
        Dims, isComplex = head[:2], head[2]
    elif len(head) == 2:
        Dims, isComplex = head, 0
    else:
        Dims, isComplex = [head[0], 1], 0
    n = int(np.prod(Dims)) * (isComplex + 1)
    vals, _ = sscanf_f('\n'.join(lines[1:]), n)
    Data = np.array(vals, dtype=float)
    if isComplex:
        Data = Data[0::2] + 1j * Data[1::2]
    return reshape(Data, Dims), np.zeros((0, 0)), np.zeros((0, 0))
