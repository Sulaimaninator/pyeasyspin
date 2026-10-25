"""Tables of 3j symbols (L1 k L2; MK1 q MK2), k = 0, 1, 2, for an orientational basis"""
import numpy as np
from .._registry import F, nout

_cache = []
_maxCache = 4


def _basis(L):
    for L1 in L:
        for MK1 in range(L1, -L1 - 1, -1):
            yield L1, MK1, L1 ** 2 + (L1 - MK1)


@nout(3)
def jjjsymbol(evenLmax=None, oddLmax=None, computeRank1=None):
    """jjjsymbol(evenLmax, oddLmax, computeRank1) -> (jjj0, jjj1, jjj2); jjj1 = empty if not computeRank1."""
    evenLmax, oddLmax = int(evenLmax), int(oddLmax)
    computeRank1 = bool(computeRank1)
    L = sorted(list(range(0, evenLmax + 1, 2)) + list(range(1, oddLmax + 1, 2)))
    nBasis = (max(L) + 1) ** 2
    for c in _cache:
        if c[0] == (evenLmax, oddLmax, computeRank1):
            return c[1]
    w3j = F.wigner3j
    jjj0 = np.zeros((nBasis, nBasis))
    for L1, MK1, i1 in _basis(L):
        for L2, MK2, i2 in _basis(L):
            if L1 != L2 and MK1 != -MK2:
                continue
            jjj0[i1, i2] = np.real(w3j(L1, 0, L2, MK1, 0, MK2))
    if computeRank1:
        jjj1 = np.zeros((nBasis, nBasis))
        for L1, MK1, i1 in _basis(L):
            for L2, MK2, i2 in _basis(L):
                if abs(-MK1 - MK2) > 1 or abs(L1 - L2) > 1:
                    continue
                jjj1[i1, i2] = np.real(w3j(L1, 1, L2, MK1, -MK2 - MK1, MK2))
    else:
        jjj1 = np.zeros((0, 0))
    jjj2 = np.zeros((nBasis, nBasis))
    for L1, MK1, i1 in _basis(L):
        for L2, MK2, i2 in _basis(L):
            if abs(-MK1 - MK2) > 2 or abs(L1 - L2) > 2:
                continue
            jjj2[i1, i2] = np.real(w3j(L1, 2, L2, MK1, -MK2 - MK1, MK2))
    out = (jjj0, jjj1, jjj2)
    if len(_cache) >= _maxCache:
        _cache.pop(0)
    _cache.append(((evenLmax, oddLmax, computeRank1), out))
    return out
