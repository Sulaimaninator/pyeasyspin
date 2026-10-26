"""Construct stick spectrum from peak data"""
import warnings
import numpy as np
from .._registry import nout
from .._mat import vec
from .._matlab import fix


@nout(-1)
def makespec(Range=None, nPoints=None, Positions=None, Amplitudes=None, nargout=None):
    """makespec(Range, nPoints, Positions, Amplitudes=ones) -> spec (1 x nPoints);
    nargout=2 -> (x, spec)."""
    Positions = vec(np.asarray(Positions, dtype=float))
    if Amplitudes is None:
        Amplitudes = np.ones(Positions.size)
    Amplitudes = vec(np.asarray(Amplitudes))
    if Amplitudes.size != Positions.size:
        raise ValueError('The position and amplitude vectors must have the same number of elements!')
    Range = vec(np.asarray(Range))
    if Range.size != 2 or np.iscomplexobj(Range):
        raise ValueError('Range must be a 2-element array [minX maxX].')
    if Range[0] >= Range[1]:
        raise ValueError('Range(2) must be larger than Range(1).')
    nPoints = int(nPoints)
    idx = (nPoints - 1) * (Positions - Range[0]) / (Range[1] - Range[0])
    idx = 1 + fix(idx)
    out = (idx < 1) | (idx > nPoints)
    if np.any(out):
        warnings.warn('Some peaks fall outside the specified range.')
        idx = idx[~out]
        Amplitudes = Amplitudes[~out]
    spec = np.zeros((1, nPoints), dtype=complex if np.iscomplexobj(Amplitudes) else float)
    np.add.at(spec[0], idx.astype(int) - 1, Amplitudes)
    if nargout == 2:
        return np.linspace(Range[0], Range[1], nPoints).reshape(1, -1), spec
    return spec
