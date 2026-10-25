"""Frequency domain axis"""
import numpy as np
from .._mat import vec


def fdaxis(*varargin):
    """fdaxis(TimeAxis) or fdaxis(dT, N) -> frequency axis (1xN row) for fftshift(fft(...))."""
    if len(varargin) == 1:
        t = vec(np.asarray(varargin[0], dtype=float))
        dT = t[1] - t[0]
        N = t.size
    elif len(varargin) == 2:
        dT = float(np.ravel(varargin[0])[0])
        N = int(np.ravel(varargin[1])[0])
    else:
        raise ValueError('Wrong number of input arguments!')
    NyquistFrequency = 1 / (2 * dT)
    UnitAxis = 2 / N * (np.arange(N) - np.fix(N / 2))
    return (NyquistFrequency * UnitAxis).reshape(1, -1)
