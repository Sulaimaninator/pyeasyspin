"""Legendre polynomials and associated Legendre polynomials"""
import numpy as np


def plegendre(*varargin):
    """plegendre(L, z), plegendre(L, M, z[, CSphase=True]) -> P_L^M(z), same shape as z."""
    nargin = len(varargin)
    if nargin < 2:
        raise ValueError('At least 2 inputs are required (L, z).')
    if nargin > 4:
        raise ValueError('At most 4 inputs are possible (L, M, z, CSphase).')
    CS = True
    if nargin == 2:
        L, M, z = varargin[0], 0, varargin[1]
    elif nargin == 3:
        L, M, z = varargin
    else:
        L, M, z, CS = varargin
    if np.size(L) != 1 or float(L) < 0 or float(L) % 1:
        raise ValueError('L (first input) must be a non-negative integer (0, 1, 2, etc).')
    L = int(round(float(L)))
    if np.size(M) != 1 or abs(float(M)) > L or float(M) % 1:
        raise ValueError('M (second input) must be an integer with -L<=M<=L.')
    M = int(round(float(M)))
    scalar = np.ndim(z) == 0
    if np.iscomplexobj(z) or np.any(np.abs(np.asarray(z)) > 1):
        raise ValueError('All values in z (third input) must satisfy -1<=z<=1.')
    z = np.asarray(z, dtype=float)
    if np.size(CS) != 1 or not isinstance(CS if not isinstance(CS, np.ndarray) else CS.item(), (bool, np.bool_)):
        raise ValueError('CSphase (4th input) must be true or false.')
    CS = bool(CS)
    negativeM = M < 0
    if negativeM:
        M = -M
    if M > 0:
        P0 = float(np.prod(np.arange(1, 2 * M, 2, dtype=float))) * np.sqrt(1 - z ** 2) ** M
    else:
        P0 = np.ones(z.shape)
    if L == M:
        y = P0
    else:
        P1 = (2 * M + 1) * z * P0
        if L == M + 1:
            y = P1
        else:
            for L_ in range(M + 2, L + 1):
                P2 = (2 * L_ - 1) / (L_ - M) * z * P1 - (L_ + M - 1) / (L_ - M) * P0
                P0 = P1
                P1 = P2
            y = P2
    if negativeM:
        y = (-1) ** M / float(np.prod(np.arange(L - M + 1, L + M + 1, dtype=float))) * y
    if CS:
        y = (-1) ** M * y
    return float(y) if scalar else y
