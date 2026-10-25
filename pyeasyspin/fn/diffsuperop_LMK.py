"""function computes the diffusion superoperator matrix in the LMK basis, for rhombic
diffusion tensors and general orienting potentials (Method 1: explicit elements; Method 2:
angular-momentum and Wigner-function operators)"""
import math
import numpy as np
import scipy.sparse as sp
from .._registry import F


def _angmomops_LMK(L, M, K):
    nBasis = L.size
    row, col, val = [], [], []
    for b1 in range(nBasis):
        L1, M1, K1 = L[b1], M[b1], K[b1]
        for b2 in range(nBasis):
            L2 = L[b2]
            if L1 != L2:
                continue
            M2 = M[b2]
            if M1 != M2:
                continue
            K2 = K[b2]
            if K1 != K2 + 1:
                continue
            row.append(b1)
            col.append(b2)
            val.append(math.sqrt(L2 * (L2 + 1) - K2 * (K2 + 1)))
    Jp = sp.csc_matrix((np.array(val, dtype=float), (row, col)), shape=(nBasis, nBasis))
    Jm = Jp.conj().T
    J2 = sp.diags(L * (L + 1.0), 0, format='csc')
    Jz = sp.diags(K.astype(float), 0, format='csc')
    return Jz, Jp, Jm, J2


def _wignerops_LMK(L, M, K, Lx, Mx, Kx):
    nBasis = L.size
    DLMK = []
    for iOp in range(len(Lx)):
        Lx_, Mx_, Kx_ = Lx[iOp], Mx[iOp], Kx[iOp]
        if Lx_ == 0:
            DLMK.append(sp.identity(nBasis, format='csc'))
            continue
        D_ = np.zeros((nBasis, nBasis))
        for b1 in range(nBasis):
            L1, M1, K1 = L[b1], M[b1], K[b1]
            for b2 in range(nBasis):
                L2, M2, K2 = L[b2], M[b2], K[b2]
                dM = M1 - M2
                dK = K1 - K2
                if Mx_ != dM:
                    continue
                if Kx_ != dK:
                    continue
                if abs(L1 - Lx_) > L2 or L2 > L1 + Lx_:
                    continue
                v = F.wigner3j([L1, Lx_, L2], [-M1, dM, M2])
                if v == 0:
                    continue
                v = F.wigner3j([L1, Lx_, L2], [-K1, dK, K2]) * v
                if v == 0:
                    continue
                D_[b1, b2] = math.sqrt((2 * L1 + 1) * (2 * L2 + 1)) * (-1.0) ** (K1 - M1) * v
        DLMK.append(sp.csc_matrix(D_))
    return DLMK


def diffsuperop_LMK(basis=None, R=None, XLMK=None, Potential=None, Method=None):
    """diffsuperop_LMK(basis (LMK), R, XLMK, Potential, Method=1) -> sparse Gamma."""
    if 'jK' in basis and np.size(basis['jK']) > 0:
        raise ValueError('This function expects an LMK basis, without jK.')
    usePotential = XLMK is not None and bool(np.any(np.asarray(Potential['lambda']) != 0))
    if usePotential:
        Lx, Mx, Kx, X = [], [], [], []
        for L_ in range(len(XLMK)):
            A = np.atleast_2d(np.asarray(XLMK[L_]))
            c, r = np.nonzero(A.T)          # find(): column-major order
            Mx += [int(v) - L_ for v in r]
            Kx += [int(v) - L_ for v in c]
            Lx += [L_] * len(c)
            X += list(A[r, c])
    if Method is None or np.size(Method) == 0:
        Method = 1
    Rv = np.asarray(R, dtype=float).ravel(order='F')
    if Rv.size == 1:
        Rd = 0.0
        Rperp = float(Rv[0])
        Rz = float(Rv[0])
        isAxial = Rd == 0
    elif Rv.size == 3:
        Rd = (Rv[0] - Rv[1]) / 4
        Rperp = (Rv[0] + Rv[1]) / 2
        Rz = Rv[2]
        isAxial = Rd == 0
    else:
        raise ValueError('Wrong number of elements in diffusion tensor.')
    L = np.asarray(basis['L'], dtype=float).ravel(order='F').astype(int)
    M = np.asarray(basis['M'], dtype=float).ravel(order='F').astype(int)
    K = np.asarray(basis['K'], dtype=float).ravel(order='F').astype(int)
    nBasis = L.size
    if Method == 1:
        if isAxial and not usePotential:
            diagonal = Rperp * (L * (L + 1.0) - K.astype(float) ** 2) + Rz * K.astype(float) ** 2
            return sp.diags(diagonal, 0, shape=(nBasis, nBasis), format='csc')
        row, col, values = [], [], []
        for b1 in range(nBasis):
            L1, M1, K1 = int(L[b1]), int(M[b1]), int(K[b1])
            for b2 in range(b1, nBasis):
                M2 = int(M[b2])
                if M1 != M2:
                    continue
                L2, K2 = int(L[b2]), int(K[b2])
                val_ = 0
                if L1 == L2:
                    LL2 = L2 * (L2 + 1)
                    if K1 == K2:
                        val_ = Rperp * (LL2 - K1 ** 2) + Rz * K1 ** 2
                    if Rd != 0:
                        if K1 == K2 + 2:
                            val_ = Rd * math.sqrt((LL2 - K2 * (K2 + 1)) * (LL2 - (K2 + 1) * (K2 + 2)))
                        elif K1 == K2 - 2:
                            val_ = Rd * math.sqrt((LL2 - K2 * (K2 - 1)) * (LL2 - (K2 - 1) * (K2 - 2)))
                if usePotential:
                    valp_ = 0
                    for p in range(len(X)):
                        if Kx[p] != K1 - K2:
                            continue
                        if Mx[p] != M1 - M2:
                            continue
                        if abs(L1 - Lx[p]) > L2 or L2 > L1 + Lx[p]:
                            continue
                        jjjxM = F.wigner3j(L1, Lx[p], L2, -M1, Mx[p], M2)
                        if jjjxM == 0:
                            continue
                        jjjxK = F.wigner3j(L1, Lx[p], L2, -K1, K1 - K2, K2)
                        if jjjxK == 0:
                            continue
                        valp_ = valp_ + jjjxM * jjjxK * X[p]
                    val_ = val_ + (-1.0) ** (K1 - M1) * math.sqrt((2 * L1 + 1) * (2 * L2 + 1)) * valp_
                if val_ == 0:
                    continue
                row.append(b1)
                col.append(b2)
                values.append(val_)
                if b1 != b2:
                    row.append(b2)
                    col.append(b1)
                    values.append(val_)
        if not values:
            raise ValueError("Unrecognized function or variable 'row'.")
        return sp.csc_matrix((np.array(values), (row, col)), shape=(nBasis, nBasis))
    elif Method == 2:
        Jz, Jp, Jm, J2 = _angmomops_LMK(L, M, K)
        Gamma = Rd * (Jp @ Jp + Jm @ Jm) + Rperp * (J2 - Jz @ Jz) + Rz * (Jz @ Jz)
        if usePotential:
            DLMK = _wignerops_LMK(L, M, K, Lx, Mx, Kx)
            for p in range(len(X)):
                Gamma = Gamma + X[p] * DLMK[p]
        return sp.csc_matrix(Gamma)
    else:
        raise ValueError('There is no method %d.' % Method)
