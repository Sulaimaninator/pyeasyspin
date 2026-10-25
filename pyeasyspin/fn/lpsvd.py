"""Prediction Singular Value Decomposition"""
import math
import numpy as np
import scipy.linalg
from .._registry import nout
from .._struct import Struct
from .._mat import vec


def _order_select(S, N, order):
    """Model order from the singular values: 'aic' or 'mdl' criterion (EasySpin's formulas)."""
    M = S.size
    crit = np.zeros(M)
    for k in range(M):
        tail = S[k:M]
        if order == 'aic':
            crit[k] = 2 * N * ((M - k) * np.log((np.sum(tail) / (M - k))) - np.sum(np.log(tail))) + 2 * k * (2 * M - k)
        else:
            crit[k] = N * ((M - k) * np.log((np.sum(tail) / (M - k))) - np.sum(np.log(tail))) \
                + k * (2 * M - k) * np.log(N) / 2
    return int(np.argmin(crit))     # = MATLAB m-1 with m the 1-based argmin


def _poles(U, S, V, m, method, dat, N, L):
    Um = U[:, :m]
    Sm = S[:m]
    Vm = V[:, :m]
    if method == 'kt':
        h = -np.conj(vec(dat)[:N - L])
        b = Vm @ (np.diag(1 / Sm) @ (Um.conj().T @ h))
        s = np.roots(np.concatenate((b[::-1], [1.0])))
        s = s[np.abs(s) < 1]
    elif method == 'ss':
        Umt = Um[1:, :]
        Umb = Um[:-1, :]
        Zp = np.linalg.solve(Umb.conj().T @ Umb, Umb.conj().T @ Umt)
        s = np.linalg.eigvals(Zp)
    elif method == 'tls':
        Umt = Um[1:, :]
        Umb = Um[:-1, :]
        _, _, Vh = np.linalg.svd(np.hstack((Umb, Umt)), full_matrices=False)
        Vu = Vh.conj().T
        X = Vu[:m, m:2 * m]
        Y = Vu[m:2 * m, m:2 * m]
        Zp = -np.linalg.solve(Y.T, X.T).T
        s = np.linalg.eigvals(Zp)
    else:
        raise ValueError(f"Unknown method '{method}'.")
    if s.size == 0:
        raise ValueError('No prediction coefficients calculated, algorithm failed to converge.')
    return s


@nout(2)
def lpsvd(data=None, time=None, method=None, order=None):
    """lpsvd(data, time, method='ss', order='mdl') -> (y, parameters): fit of a 1D signal by a sum of
    damped complex exponentials (linear prediction + SVD; method 'kt', 'ss' or 'tls'; order 'aic',
    'mdl' or a number). parameters: damping, frequency, amplitude, phase, model (callable of time)."""
    data = np.asarray(data)
    dim = data.shape if data.ndim >= 2 else (1, data.size)
    if min(dim) > 1:
        raise ValueError('Data must be a row or column vector.')
    if len(dim) > 2:
        raise ValueError('Data must be a row or column vector.')
    data = vec(data)
    time = vec(np.asarray(time, dtype=float))
    if data.size != time.size:
        raise ValueError('Time vector must be the same length as data')
    if method is None or (isinstance(method, str) and not method):
        method = 'ss'
    m = None
    if order is None:
        order = 'mdl'
    else:
        m = order
    dat = data / np.max(np.abs(data))
    N = data.size
    L = int(math.floor(0.6 * N))
    A = scipy.linalg.hankel(np.conj(dat[1:N - L + 1]), np.conj(dat[N - L:N]))
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    V = Vh.conj().T
    if isinstance(order, str):
        if order in ('aic', 'mdl'):
            m = _order_select(S, N, order)
    m = int(m)
    s = _poles(U, S, V, m, method, dat, N, L)
    s = -np.log(s.astype(complex))
    dt = time[1] - time[0]
    damp = np.real(s) / dt
    freq = np.imag(s) / (2 * math.pi * dt)
    I = damp > 0
    damp = damp[I]
    freq = freq[I]
    I = np.argsort(freq, kind='stable')
    freq = freq[I]
    damp = damp[I]
    y = np.exp(np.outer(time, -damp + 1j * 2 * math.pi * freq))
    R = np.linalg.cholesky(y.conj().T @ y).conj().T          # upper triangular, R'*R = y'*y
    a = scipy.linalg.solve_triangular(R, scipy.linalg.solve_triangular(R.conj().T, y.conj().T @ data, lower=True))
    y = y @ a
    amp = np.abs(a)
    phase = np.imag(np.log(a / amp))
    ind = phase < 0
    phase[ind] = phase[ind] + math.pi
    amp[ind] = -amp[ind]
    parameters = Struct()
    parameters.damping = damp.reshape(-1, 1)
    parameters.frequency = freq.reshape(-1, 1)
    parameters.amplitude = amp.reshape(-1, 1)
    parameters.phase = phase.reshape(-1, 1)
    dmp, frq, am, ph = damp.copy(), freq.copy(), amp.copy(), phase.copy()
    parameters.model = lambda tt: (np.exp(np.outer(vec(np.asarray(tt, dtype=float)), -dmp + 1j * 2 * math.pi * frq))
                                   @ (am * np.exp(1j * ph))).reshape(-1, 1)
    y = y.reshape(dim, order='F')
    return y, parameters
