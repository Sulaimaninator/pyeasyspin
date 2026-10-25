"""Active Spectrum (.ESR) loader."""
import numpy as np
from .._registry import nout
from .._io import textscan_lines, sscanf_f


@nout(3)
def eprload_ActiveSpectrum(FileName=None):
    """(Data, Abscissa, Parameters=[]) from an Active Spectrum text file."""
    allLines = textscan_lines(FileName)
    idx = next((i for i, L in enumerate(allLines) if L.startswith('FIELD (G)')), None)
    if idx is None:
        raise ValueError(f'Could not find start of data in file {FileName}')
    data = np.array([sscanf_f(L, 2)[0] for L in allLines[idx + 1:]], dtype=float)
    return data[:, 1:2], data[:, 0:1], np.zeros((0, 0))
