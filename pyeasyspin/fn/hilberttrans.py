"""Hilbert transform"""
import numpy as np
from .._mat import vec


def hilberttrans(y=None):
    """Analytic signal of a real row/column vector (same shape), as MATLAB hilbert()."""
    y = np.asarray(y)
    if np.iscomplexobj(y):
        raise ValueError('Input must be real.')
    dim = y.shape if y.ndim >= 2 else (1, y.size)
    if min(dim) > 1 or len(dim) > 2:
        raise ValueError('Input must be a row or column vector.')
    n = max(dim)
    h = np.zeros(n)
    if n % 2 == 0:
        idx = n // 2
        h[0] = 1
        h[idx] = 1
        h[1:idx] = 2
    else:
        idx = int(np.ceil(n / 2))
        h[0] = 1
        h[1:idx] = 2
    y_h = np.fft.ifft(np.fft.fft(vec(y)) * h)
    return y_h.reshape(y.shape, order='F')
