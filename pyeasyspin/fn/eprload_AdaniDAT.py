"""Adani spectrometer (.dat) loader."""
import numpy as np
from .._registry import nout
from .._io import textscan_lines, sscanf_f


@nout(3)
def eprload_AdaniDAT(FileName=None):
    """(Data, Abscissa, Parameters=[]) from an Adani .dat text file (decimal commas allowed)."""
    allLines = textscan_lines(FileName)
    Line1 = '======================== Parameters: ========================'
    if not allLines or not allLines[0].startswith(Line1):
        raise ValueError(f'The file {FileName} is not an Adani spectrometer file')
    Line2 = '============================================================='
    idx = next((i for i, L in enumerate(allLines) if L.startswith(Line2)), None)
    if idx is None:
        raise ValueError(f'Could not find start of data in file {FileName}')
    rows = [sscanf_f(L.replace(',', '.'), 3)[0] for L in allLines[idx + 1:]]
    data = np.zeros((len(rows), 3))
    for p, r in enumerate(rows):
        data[p, :] = r
    return data[:, 2:3], data[:, 1:2], np.zeros((0, 0))
