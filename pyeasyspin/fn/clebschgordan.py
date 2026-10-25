"""Clebsch-Gordan coefficient"""
import numpy as np
from .._registry import F
from .wigner3j import _pm1


def clebschgordan(*varargin):
    """clebschgordan(j1,j2,j,m1,m2,m), ([j1 m1],[j2 m2],[j m]) or ([j1 j2 j],[m1 m2 m])."""
    nargin = len(varargin)
    fl = lambda a: np.asarray(a, dtype=float).ravel(order='F')
    if nargin == 6:
        j1, j2, j, m1, m2, m = (float(fl(a)[0]) for a in varargin)
    elif nargin == 3:
        a, b, c = (fl(x) for x in varargin)
        j1, m1, j2, m2, j, m = a[0], a[1], b[0], b[1], c[0], c[1]
    elif nargin == 2:
        jj, mm = fl(varargin[0]), fl(varargin[1])
        j1, j2, j = jj[0], jj[1], jj[2]
        m1, m2, m = mm[0], mm[1], mm[2]
    else:
        raise ValueError('Wrong number of parameters! Need either 2, 3 or 6.')
    return _pm1(j1 - j2 + m) * np.sqrt(2 * j + 1) * F.wigner3j(j1, j2, j, m1, m2, -m)
