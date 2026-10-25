"""Transformation from ISTOs to Stevens operators"""
import numpy as np
from .._registry import F


def isto2stev(k=None):
    """isto2stev(k) -> C ((2k+1) square), Stevens = C * ISTO coefficients; rows/cols q = k..-k."""
    if np.size(k) != 1 or np.iscomplexobj(k) or float(k) < 0 or float(k) % 1:
        raise ValueError('Rank k must be a nonnegative integer (0, 1, 2, ...).')
    k = int(round(float(k)))
    J = k / 2
    n = 2 * k + 1
    T = [None] * n
    O = [None] * n
    for q in range(-k, k + 1):
        T[k - q] = np.asarray(F.isto(J, np.array([[k, q]], float)))
        O[k - q] = np.asarray(F.stev(J, np.array([[k, q]], float)))
    C = np.zeros((n, n), dtype=complex)
    for t in range(n):
        traceTT = np.trace(T[t].conj().T @ T[t])
        for o in (t, n - 1 - t):
            C[o, t] = np.trace(O[o].T @ T[t]) / traceTT
    if not np.any(C.imag):
        C = C.real
    return C
