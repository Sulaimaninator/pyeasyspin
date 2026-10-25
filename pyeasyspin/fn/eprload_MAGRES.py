"""MAGRES (.plt) loader."""
import warnings
import numpy as np
from .._registry import nout
from .._io import fgetl_lines, sscanf_f


@nout(3)
def eprload_MAGRES(FileName=None):
    """(Data, Abscissa=[], Parameters=[]) from a MAGRES .PLT file."""
    try:
        lines = fgetl_lines(FileName)
    except OSError:
        raise ValueError('Could not open ' + FileName)
    nx = 0
    for L in lines:          # findtagsMAGRES: the last line whose first token is DATA wins
        ws = [i for i, c in enumerate(L) if c.isspace()]
        if ws and ws[0] > 0 and L[:ws[0]] == 'DATA':
            try:
                nx = float(L[ws[0]:].strip())
            except ValueError:
                nx = float('nan')
    if not nx or nx != nx:
        raise ValueError('Unable to determine number of x points in PLT file.')
    nx = int(nx)
    vals, _ = sscanf_f('\n'.join(lines[3:]), nx)
    if len(vals) < nx:
        warnings.warn('Could not read entire data set from PLT file.')
    Data = np.array(vals, dtype=float).reshape(-1, 1)
    return Data, np.zeros((0, 0)), np.zeros((0, 0))
