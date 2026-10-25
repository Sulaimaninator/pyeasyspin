"""basis with basis functions and ordering as used in the Freed program"""
import numpy as np
from .._struct import Struct


def _r(a, b, step=1):
    """MATLAB a:step:b for integer-valued a, b, step."""
    return range(int(round(a)), int(round(b)) + (1 if step > 0 else -1), int(step))


def chili_basisbuild(Basis=None, Sys=None):
    """chili_basisbuild(Basis, Sys) -> Basis with the K-symmetrized orientational/spin basis
    (fields L, jK, K, M, pS, qS[, pI1, qI1[, pI2, qI2]] as column vectors), in Freed's ordering."""
    I = np.asarray(Sys['I'], dtype=float).ravel(order='F')
    nNuclei = I.size
    if nNuclei >= 1:
        I1 = float(I[0])
    if nNuclei >= 2:
        I2 = float(I[1])
    DirTilt = bool(np.any(Basis['DirTilt']))
    evenLmax = Basis['evenLmax']
    oddLmax = Basis['oddLmax']
    jKmin = Basis['jKmin']
    Kmax = Basis['Kmax']
    deltaK = int(bool(Basis['evenK'])) + 1
    Mmax = Basis['Mmax']
    pSmin = Basis['pSmin']
    pImax = np.asarray(Basis['pImax'], dtype=float).ravel(order='F')
    if nNuclei == 1:
        pI1max = pImax
    elif nNuclei == 2:
        pI1max = pImax[0]
        pI2max = pImax[1]
    if nNuclei >= 1:
        pI1max = float(np.ravel(pI1max)[0])
    MpSymm = bool(Basis['MpSymm'])

    rows = []
    for L in _r(0, evenLmax):
        Lparity = 1 - 2 * (L % 2)
        evenL = Lparity == 1
        if (not evenL) and L > oddLmax:
            continue
        for jK in _r(jKmin, 1, 2):
            for K in _r(0, min(L, Kmax), deltaK):
                if K == 0 and Lparity != jK:
                    continue
                Mmx = min(L, Mmax)
                for M in _r(-Mmx, Mmx):
                    for pS in _r(pSmin, 1):
                        qSmx = 1 - abs(pS)
                        for qS in _r(-qSmx, qSmx, 2):
                            if nNuclei == 0:
                                if MpSymm and not DirTilt and (pS - 1) != M:
                                    continue
                                rows.append((L, jK, K, M, pS, qS))
                            elif nNuclei == 1:
                                for pI1 in _r(-pI1max, pI1max):
                                    if MpSymm and not DirTilt and pI1 + pS - 1 != M:
                                        continue
                                    qI1max = 2 * I1 - abs(pI1)
                                    for qI1 in _r(-qI1max, qI1max, 2):
                                        rows.append((L, jK, K, M, pS, qS, pI1, qI1))
                            elif nNuclei == 2:
                                for pI1 in _r(-pI1max, pI1max):
                                    qI1max = 2 * I1 - abs(pI1)
                                    for qI1 in _r(-qI1max, qI1max, 2):
                                        for pI2 in _r(-pI2max, pI2max):
                                            if MpSymm and not DirTilt and pI1 + pI2 + pS - 1 != M:
                                                continue
                                            qI2max = 2 * I2 - abs(pI2)
                                            for qI2 in _r(-qI2max, qI2max, 2):
                                                rows.append((L, jK, K, M, pS, qS, pI1, qI1, pI2, qI2))
    ncol = 4 + 2 + 2 * nNuclei
    Indices = np.array(rows, dtype=float).reshape(-1, ncol)
    Basis = Struct(Basis)
    names = ['L', 'jK', 'K', 'M', 'pS', 'qS']
    if nNuclei >= 1:
        names += ['pI1', 'qI1']
        if nNuclei >= 2:
            names += ['pI2', 'qI2']
    for k, nm in enumerate(names):
        Basis[nm] = Indices[:, k:k + 1].copy()
    return Basis
