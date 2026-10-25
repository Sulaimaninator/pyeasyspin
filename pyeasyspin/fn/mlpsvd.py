"""Linear Prediction Singular Value Decomposition down a single dimension"""
import math
import numpy as np
import scipy.linalg
from .._registry import nout
from .._struct import Struct
from .._mat import vec
from .lpsvd import _order_select, _poles


@nout(2)
def mlpsvd(data=None, time=None, method=None, order=None, method2d=None):
    """mlpsvd(data, time, method='ss', order='mdl', method2d='sum') -> (y, parameters): lpsvd fit of
    all rows (or columns) of a 2D array with common damping/frequency poles (method2d 'sum' or 'stack');
    amplitudes and phases are per trace."""
    data = np.atleast_2d(np.asarray(data))
    time = vec(np.asarray(time, dtype=float))
    dim = data.shape
    idx = [i + 1 for i, d in enumerate(dim) if d == time.size]
    if not idx:
        raise ValueError('Time vector must be the same length as data')
    idx = idx[0]
    time = time.reshape(1, -1)
    if idx == 1:
        data = data.T
        dim = data.shape
    if method is None or (isinstance(method, str) and not method):
        method = 'ss'
    m = None
    if order is None or (isinstance(order, (str, np.ndarray, list)) and len(order) == 0):
        order = 'mdl'
    else:
        m = order
    if method2d is None or (isinstance(method2d, str) and not method2d):
        method2d = 'sum'
    N = dim[1]
    L = int(math.floor(0.6 * N))
    if method2d == 'sum':
        dat = np.sum(data, axis=0).reshape(1, -1)
        dat = dat / np.max(np.abs(dat))
        A = scipy.linalg.hankel(np.conj(dat[0, 1:N - L + 1]), np.conj(dat[0, N - L:N]))
    elif method2d == 'stack':
        dat = data / np.max(np.abs(data))
        A = np.zeros((dim[0] * (N - L), L), dtype=dat.dtype)
        for i in range(dim[0]):
            H = scipy.linalg.hankel(np.conj(dat[i, 1:N - L + 1]), np.conj(dat[i, N - L:N]))
            A[i * (N - L):(i + 1) * (N - L), :] = H
    else:
        raise ValueError(f"Unknown method2d '{method2d}'.")
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    V = Vh.conj().T
    if isinstance(order, str):
        if order in ('aic', 'mdl'):
            m = _order_select(S, N, order)
    m = int(m)
    s = _poles(U, S, V, m, method, dat, N, L)
    s = -np.log(s.astype(complex))
    tv = time.ravel()
    dt = tv[1] - tv[0]
    damp = np.real(s) / dt
    freq = np.imag(s) / (2 * math.pi * dt)
    I = damp > 0
    damp = damp[I]
    freq = freq[I]
    I = np.argsort(freq, kind='stable')
    freq = freq[I]
    damp = damp[I]
    y = np.exp(np.outer(-damp + 1j * 2 * math.pi * freq, tv))
    a = np.linalg.solve((y @ y.conj().T).T, (data @ y.conj().T).T).T
    y = a @ y
    amp = np.abs(a)
    phase = np.imag(np.log(a / amp))
    ind = phase < 0
    phase[ind] = phase[ind] + math.pi
    amp[ind] = -amp[ind]
    parameters = Struct()
    parameters.damping = damp.reshape(-1, 1)
    parameters.frequency = freq.reshape(-1, 1)
    parameters.amplitude = amp
    parameters.phase = phase
    dmp, frq, am, ph = damp.copy(), freq.copy(), amp.copy(), phase.copy()
    parameters.model = lambda tt: (am * np.exp(1j * ph)) @ np.exp(np.outer(-dmp + 1j * 2 * math.pi * frq,
                                                                           vec(np.asarray(tt, dtype=float))))
    if idx == 1:
        y = y.T
    return y, parameters
