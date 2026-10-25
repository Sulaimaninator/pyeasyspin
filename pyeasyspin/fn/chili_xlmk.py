"""Compute the coefficients X^L_MK of the potential-dependent part of the diffusion operator"""
import math
import numpy as np
from .._registry import F


def _col(x):
    return np.asarray(x).ravel(order='F')


def chili_xlmk(Potential=None, R=None):
    """chili_xlmk(Potential (struct L, M, K, lambda), R (1 or 3 diffusion principal values)) ->
    X: list (cell) of (2L+1)x(2L+1) matrices X{L+1}(M+L+1, K+L+1), L = 0..2*max(Lpot);
    [] if there is no potential."""
    lam_in = _col(Potential['lambda'])
    if lam_in.size == 0 or np.all(lam_in == 0):
        return []
    Lpot = _col(Potential['L']).astype(float)
    Mpot = _col(Potential['M']).astype(float)
    Kpot = _col(Potential['K']).astype(float)
    lam = lam_in.copy()
    rmv = lam == 0
    if np.any(rmv):
        Lpot, Mpot, Kpot, lam = Lpot[~rmv], Mpot[~rmv], Kpot[~rmv], lam[~rmv]
    idx = (Mpot != 0) | (Kpot != 0)
    sgn = np.array([(-1.0) ** int(round(v)) for v in (Kpot[idx] - Mpot[idx])])
    lam = np.concatenate([lam, np.conj(lam[idx]) * sgn])
    Lpot = np.concatenate([Lpot, Lpot[idx]])
    Mpot = np.concatenate([Mpot, -Mpot[idx]])
    Kpot = np.concatenate([Kpot, -Kpot[idx]])
    nPotentialTerms = lam.size
    maxLpot = int(max(Lpot))
    allMzero = bool(np.all(Mpot == 0))
    cplx = np.iscomplexobj(lam)
    dt = complex if cplx else float
    lam_ = [np.zeros((2 * L + 1, 2 * L + 1), dtype=dt) for L in range(maxLpot + 1)]
    for p in range(nPotentialTerms):
        L1_ = int(Lpot[p]) + 1
        lam_[L1_ - 1][int(Mpot[p]) + L1_ - 1, int(Kpot[p]) + L1_ - 1] = lam[p]

    def lamf(L, M, K):
        return lam_[L][M + L, K + L]

    Rv = _col(R).astype(float)
    if Rv.size == 1:
        Rv = np.array([Rv[0]] * 3)
    elif Rv.size != 3:
        raise ValueError('Provide 3 principal values of diffusion tensor as second input.')
    Rz = Rv[2]
    Rp = (Rv[0] + Rv[1]) / 2
    Rd = (Rv[0] - Rv[1]) / 4
    maxLx = maxLpot * 2
    X = [np.zeros((2 * L + 1, 2 * L + 1), dtype=dt) for L in range(maxLx + 1)]

    def cm(L, K):
        return math.sqrt(L * (L + 1) - K * (K - 1))

    def cp(L, K):
        return math.sqrt(L * (L + 1) - K * (K + 1))

    memo = {}

    def w3j(*a):
        v = memo.get(a)
        if v is None:
            v = memo[a] = F.wigner3j(*a)
        return v

    Lp = [int(v) for v in Lpot]
    Mp = [int(v) for v in Mpot]
    Kp = [int(v) for v in Kpot]
    for L in range(maxLx + 1):
        Mrange = [0] if allMzero else range(-L, L + 1)
        for M in Mrange:
            for K in range(-L, L + 1):
                A = 0
                if L <= maxLpot:
                    if Rd != 0:
                        if K + 2 <= L:
                            A = A + Rd * cm(L, K + 1) * cm(L, K + 2) * lamf(L, M, K + 2)
                        if K - 2 >= -L:
                            A = A + Rd * cp(L, K - 1) * cp(L, K - 2) * lamf(L, M, K - 2)
                    A = A + lamf(L, M, K) * (Rp * (L * (L + 1) - K ** 2) + Rz * K ** 2)
                B = 0
                for p1 in range(nPotentialTerms):
                    lam1 = lam[p1]
                    if lam1 == 0:
                        continue
                    L1, M1, K1 = Lp[p1], Mp[p1], Kp[p1]
                    for p2 in range(nPotentialTerms):
                        lam2 = lam[p2]
                        if lam2 == 0:
                            continue
                        L2, M2, K2 = Lp[p2], Mp[p2], Kp[p2]
                        if M1 - M + M2 != 0:
                            continue
                        if abs(L1 - L2) > L or L > L1 + L2:
                            continue
                        if M1 == 0 and M2 == 0 and M == 0:
                            if (L1 + L2 + L) % 2:
                                continue
                        B_ = 0
                        if Rd != 0:
                            if K1 - K + K2 + 2 == 0 and abs(K1 + 1) <= L1 and abs(K2 + 1) <= L2:
                                cpp = cp(L1, K1) * cp(L2, K2)
                                if cpp != 0:
                                    B_ = B_ + Rd * cpp * w3j(L1, L, L2, K1 + 1, -K, K2 + 1)
                            if K1 - K + K2 - 2 == 0 and abs(K1 - 1) <= L1 and abs(K2 - 1) <= L2:
                                cmm = cm(L1, K1) * cm(L2, K2)
                                if cmm != 0:
                                    B_ = B_ + Rd * cmm * w3j(L1, L, L2, K1 - 1, -K, K2 - 1)
                        if K1 - K + K2 == 0:
                            if Rz != 0 and K1 != 0 and K2 != 0:
                                B_ = B_ + Rz * K1 * K2 * w3j(L1, L, L2, K1, -K, K2)
                            if Rp != 0:
                                if abs(K1 + 1) <= L1 and abs(K2 - 1) <= L2:
                                    cpm = cp(L1, K1) * cm(L2, K2)
                                    if cpm != 0:
                                        B_ = B_ + Rp * cpm * w3j(L1, L, L2, K1 + 1, -K, K2 - 1)
                        if B_ == 0:
                            continue
                        B = B + lam1 * lam2 * w3j(L1, L, L2, M1, -M, M2) * B_
                X_ = -1 / 2 * A - (2 * L + 1) / 4 * (-1) ** (K - M) * B
                X[L][M + L, K + L] = X_
    return X
