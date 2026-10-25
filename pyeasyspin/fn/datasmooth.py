"""Moving average smoothing and differentiation"""
import numpy as np
from scipy.signal import lfilter
from scipy.special import comb


def datasmooth(y=None, m=None, Method=None, PolyOrder=None, Derivative=None):
    """datasmooth(y, m=3, Method='binom', PolyOrder=2, Derivative=0) -> filtered y
    (methods 'flat', 'binom', 'savgol'; window 2m+1; ends padded with the end values)."""
    nargin = 1 + sum(a is not None for a in (m, Method, PolyOrder, Derivative))
    if m is None:
        m = 3
    if Method is None:
        Method = 'binom'
    if PolyOrder is None:
        PolyOrder = 2
    if Derivative is None:
        Derivative = 0
    if np.iscomplexobj(m) or np.size(m) != 1 or not np.isfinite(m) or m < 0 or np.mod(m, 1):
        raise ValueError('m (second parameter) must be a positive integer!')
    if m == 0:
        return y
    if PolyOrder < Derivative:
        raise ValueError('Polynomial order must not be smaller than the derivative index!')
    if PolyOrder < 1 or np.mod(PolyOrder, 1) or np.size(PolyOrder) != 1:
        raise ValueError('Polynomial order p must be a positive integer!')
    m = int(m)
    n = 2 * m + 1
    if Method == 'flat':
        if nargin != 3:
            raise ValueError('Wrong number of input arguments!')
        Weights = np.ones(n) / n
    elif Method == 'binom':
        if nargin < 2 or nargin > 3:
            raise ValueError('Wrong number of input arguments!')
        Weights = comb(n - 1, np.arange(n), exact=False) / 2 ** (n - 1)
    elif Method == 'savgol':
        p = int(PolyOrder)
        X = np.arange(-m, m + 1, dtype=float).reshape(-1, 1) ** np.arange(p + 1)
        Weights = (-1) ** int(Derivative) * np.linalg.pinv(X)[int(Derivative), :]
    else:
        raise ValueError('Unknown value for third argument!')
    y = np.asarray(y)
    if y.ndim < 2:
        y = y.reshape(1, -1)
    RowVec = y.shape[0] == 1 and y.ndim == 2
    if RowVec:
        y = y.T
    y_expanded = np.concatenate([np.repeat(y[:1], m, axis=0), y, np.repeat(y[-1:], m + 1, axis=0)], axis=0)
    yf = lfilter(Weights, 1, y_expanded, axis=0)
    yf = yf[n - 1:-1]
    if RowVec:
        yf = yf.T
    return yf
