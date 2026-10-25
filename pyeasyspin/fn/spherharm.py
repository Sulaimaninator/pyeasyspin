"""Complex- and real-valued spherical harmonics"""
import numpy as np
from .._registry import F


def _normfactor(L, M):
    p = float(np.prod(np.arange(L - abs(M) + 1, L + abs(M) + 1, dtype=float)))
    if M > 0:
        p = 1 / p
    return np.sqrt((2 * L + 1) / (4 * np.pi) * p)


def spherharm(L=None, M=None, theta=None, phi=None, Type=None):
    """spherharm(L, M, theta, phi[, 'c'|'r']) -> Y_LM(theta, phi)."""
    if phi is None:
        raise ValueError('At least four inputs are required: L, M, theta, phi.')
    if Type is None:
        Type = 'c'
    if np.size(L) != 1 or float(L) < 0 or float(L) % 1:
        raise ValueError('L (first input) must be a non-negative integer (0, 1, 2, etc).')
    L = int(round(float(L)))
    if np.size(M) != 1 or abs(float(M)) > L or float(M) % 1:
        raise ValueError('M (second input) must be an integer between -L and L.')
    M = int(round(float(M)))
    if np.size(theta) != np.size(phi):
        raise ValueError('theta and phi (3rd and 4th input) must have the same size.')
    if np.iscomplexobj(theta):
        raise ValueError('theta (3rd input) must be real-valued.')
    if np.iscomplexobj(phi):
        raise ValueError('phi (4th input) must be real-valued.')
    if not isinstance(Type, str) or len(Type) != 1 or Type not in 'rc':
        raise ValueError("If given, fourth input must be 'r' or 'c'.")
    scalar = np.ndim(theta) == 0
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if Type == 'r':
        N = _normfactor(L, abs(M))
        P = np.asarray(F.plegendre(L, abs(M), np.cos(theta), False))
        if M > 0:
            y = np.sqrt(2) * N * P * np.cos(M * phi)
        elif M < 0:
            y = np.sqrt(2) * N * P * np.sin(abs(M) * phi)
        else:
            y = N * P
    else:
        y = _normfactor(L, M) * np.asarray(F.plegendre(L, M, np.cos(theta), True)) * np.exp(1j * M * phi)
    if scalar:
        y = y.item() if isinstance(y, np.ndarray) else y
    return y
