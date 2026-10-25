"""Calculation of diffusion operator matrix for general potential, in the LMK or the
K-symmetrized LMKjK orientational basis"""
import cmath
import math
import numpy as np
import scipy.sparse as sp
from .._registry import F


def _msqrt(x):
    """MATLAB sqrt: complex for negative input."""
    return math.sqrt(x) if x >= 0 else cmath.sqrt(x)


def _sqrtjk(jK):
    return 1.0 if jK >= 0 else 1j * math.sqrt(-jK)


def _real_if(v):
    """MATLAB drops an all-zero imaginary part of arithmetic results."""
    return v.real if isinstance(v, complex) and v.imag == 0 else v


def diffsuperop(basis=None, R=None, XLMK=None, Potential=None):
    """diffsuperop(basis, R, XLMK, Potential) -> sparse diffusion superoperator Gamma (rotational
    diffusion tensor principal values R, potential coefficients XLMK from chili_xlmk)."""
    includePotential = XLMK is not None and (len(XLMK) > 0 if isinstance(XLMK, (list, tuple))
                                             else np.size(XLMK) > 0)
    allMzero = Potential is not None and bool(np.all(np.asarray(Potential['M']) == 0))
    useKSym = 'jK' in basis and np.size(basis['jK']) > 0 and bool(np.any(basis['jK']))
    L = np.asarray(basis['L'], dtype=float).ravel(order='F').astype(int)
    M = np.asarray(basis['M'], dtype=float).ravel(order='F').astype(int)
    K = np.asarray(basis['K'], dtype=float).ravel(order='F').astype(int)
    nBasis = L.size
    if useKSym:
        jK = np.asarray(basis['jK'], dtype=float).ravel(order='F').astype(int)
    Rv = np.asarray(R, dtype=float).ravel(order='F')
    if Rv.size == 1:
        Rv = np.array([Rv[0]] * 3)
    elif Rv.size != 3:
        raise ValueError('Provide 3 principal values of diffusion tensor as second input.')
    Rz = Rv[2]
    Rp = (Rv[0] + Rv[1]) / 2
    Rd = (Rv[0] - Rv[1]) / 4
    if not includePotential and Rd == 0:
        diagonal = Rp * (L * (L + 1.0) - K.astype(float) ** 2) + Rz * K.astype(float) ** 2
        return sp.diags(diagonal, 0, shape=(nBasis, nBasis), format='csc')

    def Np(L, K):
        return _msqrt((L * (L + 1) - K * (K + 1)) * (L * (L + 1) - (K + 1) * (K + 2)))

    def Nm(L, K):
        return _msqrt((L * (L + 1) - K * (K - 1)) * (L * (L + 1) - (K - 1) * (K - 2)))

    row, col, values = [], [], []
    for b1 in range(nBasis):
        L1, M1, K1 = L[b1], M[b1], K[b1]
        if useKSym:
            jK1 = jK[b1]
        for b2 in range(b1, nBasis):
            L2 = L[b2]
            if L1 != L2:
                continue
            M2 = M[b2]
            if M1 != M2:
                continue
            if useKSym:
                jK2 = jK[b2]
                if jK1 != jK2:
                    continue
            K2 = K[b2]
            if K1 != K2 and K1 != K2 + 2 and K1 != K2 - 2:
                continue
            if K1 == K2:
                val_ = Rp * L2 * (L2 + 1) + (Rz - Rp) * K2 ** 2
            else:
                val_ = 0
            if useKSym:
                val2_ = Np(L2, K2) * (K1 == K2 + 2) + Nm(L2, K2) * ((K1 == K2 - 2) + jK1 * (-1.0) ** (L1 + K1) * (-K1 == K2 - 2))
                val_ = val_ + Rd * val2_ / math.sqrt((1 + (K1 == 0)) * (1 + (K2 == 0)))
            else:
                val_ = val_ + Rd * (Np(L2, K2) * (K1 == K2 + 2) + Nm(L2, K2) * (K1 == K2 - 2))
            val_ = _real_if(complex(val_)) if isinstance(val_, complex) else val_
            if val_ == 0:
                continue
            row.append(b1)
            col.append(b2)
            values.append(val_)
            if b1 != b2:
                row.append(b2)
                col.append(b1)
                values.append(np.conj(val_))
    if not values:
        raise ValueError("Unrecognized function or variable 'row'.")
    Gamma = sp.csc_matrix((np.array(values), (row, col)), shape=(nBasis, nBasis))
    if not includePotential:
        return Gamma

    Xc = [np.atleast_2d(np.asarray(x)) for x in XLMK]
    Lxmax = len(Xc) - 1

    def X(L, M, K):
        return Xc[L][M + L, K + L]

    memo = {}

    def w3j(*a):
        v = memo.get(a)
        if v is None:
            v = memo[a] = F.wigner3j(*a)
        return v

    rowp, colp, valp = [], [], []
    for b1 in range(nBasis):
        L1, M1, K1 = int(L[b1]), int(M[b1]), int(K[b1])
        if useKSym:
            jK1 = int(jK[b1])
        for b2 in range(b1, nBasis):
            L2, M2 = int(L[b2]), int(M[b2])
            if allMzero:
                if M1 != M2:
                    continue
            K2 = int(K[b2])
            if useKSym:
                jK2 = int(jK[b2])
            if useKSym:
                val_ = 0
                for Lx in range(abs(L1 - L2), min(Lxmax, L1 + L2) + 1):
                    if abs(M1 - M2) > Lx:
                        continue
                    v = 0
                    if abs(K1 - K2) <= Lx:
                        X1_ = X(Lx, M1 - M2, K1 - K2) + X(Lx, M1 - M2, -K1 + K2) * jK1 * jK2 * (-1.0) ** (Lx + K1 + K2)
                        if X1_ != 0:
                            v = v + X1_ * w3j(L1, Lx, L2, -K1, K1 - K2, K2)
                    if abs(K1 + K2) <= Lx:
                        X2_ = X(Lx, M1 - M2, -K1 - K2) * jK1 * (-1.0) ** (K1 + Lx + L2) \
                            + X(Lx, M1 - M2, K1 + K2) * jK2 * (-1.0) ** (L2 + K2)
                        if X2_ != 0:
                            v = v + X2_ * w3j(L1, Lx, L2, -K1, K1 + K2, -K2)
                    if v == 0:
                        continue
                    val_ = val_ + w3j(L1, Lx, L2, -M1, M1 - M2, M2) * v
                if val_ == 0:
                    continue
                prefactorjK = np.conj(_sqrtjk(jK1)) * _sqrtjk(jK2) / 2 / math.sqrt((1 + (K1 == 0)) * (1 + (K2 == 0)))
                prefactorL = math.sqrt((2 * L1 + 1) * (2 * L2 + 1))
                val_ = (-1.0) ** (K1 - M1) * prefactorL * prefactorjK * val_
            else:
                val_ = 0
                for Lx in range(abs(L1 - L2), min(Lxmax, L1 + L2) + 1):
                    if abs(M1 - M2) > Lx:
                        continue
                    if abs(K1 - K2) > Lx:
                        continue
                    X_ = X(Lx, M1 - M2, K1 - K2)
                    if X_ == 0:
                        continue
                    threej_M = w3j(L1, Lx, L2, -M1, M1 - M2, M2)
                    if threej_M == 0:
                        continue
                    threej_K = w3j(L1, Lx, L2, -K1, K1 - K2, K2)
                    if threej_K == 0:
                        continue
                    val_ = val_ + X_ * threej_M * threej_K
                val_ = (-1.0) ** (K1 - M1) * math.sqrt((2 * L1 + 1) * (2 * L2 + 1)) * val_
            if val_ == 0:
                continue
            rowp.append(b1)
            colp.append(b2)
            valp.append(val_)
            if b1 != b2:
                rowp.append(b2)
                colp.append(b1)
                valp.append(np.conj(val_))
    if not valp:
        raise ValueError("Unrecognized function or variable 'rowp'.")
    return sp.csc_matrix(Gamma + sp.csc_matrix((np.array(valp), (rowp, colp)), shape=(nBasis, nBasis)))
