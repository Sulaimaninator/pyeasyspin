"""Starting vector for chili: the equilibrium-weighted P+ coherence (orientational part nonzero
only for even L and K, M=0, jK=1; potential: non-zero value only for L==K==0 without potential)"""
import math
import numpy as np
import scipy.sparse as sp
import scipy.special
from .._registry import F
from .chili_eqpopvec import _integral


def _orifun(z, L, K, lam):
    p = lam.size
    A = 0
    if lam[0]:
        A = A + lam[0] / 2 * F.plegendre(2, 0, z)
    if p >= 3 and lam[2]:
        A = A + lam[2] / 2 * F.plegendre(4, 0, z)
    B = 0
    if p >= 2 and lam[1]:
        B = B + lam[1] / (2 * math.sqrt(6)) * F.plegendre(2, 2, z)
    if p == 4 and lam[3]:
        B = B + lam[3] / (6 * math.sqrt(10)) * F.plegendre(4, 2, z)
    return F.plegendre(L, K, z) * np.exp(A) * scipy.special.iv(K / 2, B) * (2 * math.pi)


def chili_startingvector(Basis=None, lambda_=None):
    """chili_startingvector(Basis, lambda) -> sparse unit column vector over the orientational
    basis; lambda = [] or the 4 coefficients of the L=2,4 / K=0,2 (M=0) potential terms."""
    lam = np.asarray([] if lambda_ is None else lambda_, dtype=float).ravel(order='F')
    if lam.size != 0 and lam.size != 4:
        raise ValueError('This functions works only for potentials with M=0, L=2,4, and K=0,2.')
    isPotential = bool(np.any(lam))
    maxKp = 2
    isNuc1 = 'pI1' in Basis
    isNuc2 = 'pI2' in Basis
    BL = np.asarray(Basis['L'], dtype=float).ravel(order='F')
    BK = np.asarray(Basis['K'], dtype=float).ravel(order='F')
    BM = np.asarray(Basis['M'], dtype=float).ravel(order='F')
    BjK = np.asarray(Basis['jK'], dtype=float).ravel(order='F')
    BpS = np.asarray(Basis['pS'], dtype=float).ravel(order='F')
    BpI1 = np.asarray(Basis['pI1'], dtype=float).ravel(order='F') if isNuc1 else None
    BpI2 = np.asarray(Basis['pI2'], dtype=float).ravel(order='F') if isNuc2 else None
    nBasis = BL.size
    stvec = np.zeros(nBasis)
    absTol = 1e-8
    cacheval = math.nan
    cacheLK = (math.nan, math.nan)
    for b in range(nBasis):
        L = BL[b]
        K = BK[b]
        if L % 2 or K % 2:
            continue
        if BM[b] != 0:
            continue
        if BjK[b] != 1:
            continue
        if abs(BpS[b]) != 1:
            continue
        if isNuc1 and BpI1[b] != 0:
            continue
        if isNuc2 and BpI2[b] != 0:
            continue
        if not isPotential:
            if L != 0 or K != 0:
                continue
            stvec[b] = 1
        else:
            if maxKp == 0 and K != 0:
                continue
            if (L, K) == cacheLK:
                value = cacheval
            else:
                Li, Ki = int(L), int(K)
                value = _integral(lambda z: _orifun(z, Li, Ki, lam), 0, 1, AbsTol=absTol)
                value = value * math.sqrt((2 * L + 1) / float(np.prod(np.arange(L - K + 1, L + K + 1))))
                if K == 0:
                    value = value / math.sqrt(2)
                cacheval = value
                cacheLK = (L, K)
            stvec[b] = value
    stvec = stvec / np.linalg.norm(stvec)
    return sp.csc_matrix(stvec.reshape(-1, 1))
