"""Derivative of a vector or matrix"""
import numpy as np
from .._mat import vec


def deriv(*varargin):
    """deriv(y) or deriv(x, y) -> dy/dx along columns (along the row for a row vector);
    central differences inside, one-sided at the ends."""
    if len(varargin) == 1:
        y, x = varargin[0], None
    elif len(varargin) == 2:
        x, y = varargin
    else:
        raise ValueError('Wrong number of input arguments!')
    y = np.asarray(y)
    if y.ndim < 2:
        y = y.reshape(1, -1)
    RowVector = y.size == y.shape[1]
    if RowVector:
        y = vec(y).reshape(-1, 1)
    if x is None or np.size(x) == 0:
        x = np.arange(1, y.shape[0] + 1, dtype=float)
    dx = np.diff(vec(np.asarray(x, dtype=float))).reshape(-1, 1)
    d = np.diff(y, axis=0) / dx
    m = d.shape[0]
    dydx = (d[np.r_[0, 0:m], :] + d[np.r_[0:m, m - 1], :]) / 2
    if RowVector:
        dydx = dydx.T
    return dydx
