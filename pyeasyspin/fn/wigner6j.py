"""Wigner 6j symbol"""
import math
import numpy as np


def _bino(a, b):
    return float(np.prod(np.arange(a - b + 1, a + 0.5) / np.arange(1, b + 0.5)))


def _fac(a):
    return float(np.prod(np.arange(1, a + 0.5)))


def _tri(a, b, c):
    return (a + b >= c) and (b + c >= a) and (c + a >= b)


def wigner6j(*varargin):
    """wigner6j(j1..j6), ([j1 j4],[j2 j5],[j3 j6]), ([j1 j2 j3],[j4 j5 j6]), (6-vector) or (n x 6 array -> 1xn)."""
    nargin = len(varargin)
    fl = lambda a: np.asarray(a, dtype=float).ravel(order='F')
    if nargin == 6:
        j1, j2, j3, j4, j5, j6 = (fl(a)[0] for a in varargin)
    elif nargin == 3:
        a, b, c = (fl(x) for x in varargin)
        j1, j4, j2, j5, j3, j6 = a[0], a[1], b[0], b[1], c[0], c[1]
    elif nargin == 2:
        j, J = fl(varargin[0]), fl(varargin[1])
        j1, j2, j3, j4, j5, j6 = j[0], j[1], j[2], J[0], J[1], J[2]
    elif nargin == 1:
        j = np.asarray(varargin[0], dtype=float)
        if j.size == 6:
            j1, j2, j3, j4, j5, j6 = j.ravel(order='F')
        else:
            if j.ndim == 2 and j.shape[1] == 6:
                return np.array([[wigner6j(j[k, :]) for k in range(j.shape[0])]])
            raise ValueError('Wrong size of input array, must be n x 6.')
    else:
        raise ValueError('Wrong number of parameters!')
    js = [float(x) for x in (j1, j2, j3, j4, j5, j6)]
    j1, j2, j3, j4, j5, j6 = js
    if any(2 * x != math.floor(2 * x) for x in js):
        raise ValueError('All parameters must be integers or half-integers.')
    if not (_tri(j1, j2, j3) and _tri(j1, j5, j6) and _tri(j4, j2, j6) and _tri(j4, j5, j3)):
        return 0.0
    A1, A2, A3 = sorted([j1 + j2 + j4 + j5, j1 + j3 + j4 + j6, j2 + j3 + j5 + j6])
    B1, B2, B3, B4 = sorted([j1 + j2 + j3, j1 + j5 + j6, j2 + j4 + j6, j3 + j4 + j5])
    f = _fac
    pre = (f(A1 - B2) * f(A1 - B3) * f(A1 - B4) / f(A1 - B1) / f(B1 + 1)
           * f(A2 - B1) * f(A2 - B3) * f(A2 - B4) / f(A2 - B2) / f(B2 + 1)
           * f(A3 - B1) * f(A3 - B2) * f(A3 - B4) / f(A3 - B3) / f(B3 + 1))
    pre = np.sqrt(pre * f(B4 + 1))
    sumfactor = (-1.0) ** int(B4) * _bino(A1 - B1, A1 - B4) * _bino(A2 - B2, A2 - B4) * _bino(A3 - B3, A3 - B4)
    pre = pre * sumfactor
    if B4 >= A1:
        return float(pre)
    t = 1.0
    s = t
    for n in np.arange(B4 + 1, A1 + 0.5):
        t = -t * (n + 1) / (n - B4) * (A1 - n + 1) / (n - B1) * (A2 - n + 1) / (n - B2) * (A3 - n + 1) / (n - B3)
        s = s + t
    return float(pre * s)
