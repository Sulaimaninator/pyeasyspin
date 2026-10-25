"""Relaxation superoperator (T1 population exchange, T2 coherence decay) in Liouville space."""
import numpy as np

from .._mat import vec


def s_relaxationsuperoperator(System=None):
    """Gamma (N^2 x N^2, N = Hilbert dimension) from System.Spins, System.T1, System.T2 (us).
    Scalars apply to all transitions; matrices are per transition (T1 lower triangle = upward rates,
    T2 upper triangle mirrored). Zero times mean no relaxation (1e10)."""
    T1in = np.atleast_2d(np.asarray(System['T1'], dtype=float))
    T2in = np.atleast_2d(np.asarray(System['T2'], dtype=float))
    t11, t12 = T1in.shape
    t21, t22 = T2in.shape
    HD = int(np.prod(vec(np.asarray(System['Spins'], dtype=float)) * 2 + 1))
    T1ud = 0
    if t11 == 1 and t12 == 1:
        T1 = T1in[0, 0] * np.ones((HD, HD))
    else:
        T1 = T1in.copy()
        if np.any(np.tril(T1)):
            T1ud = 1
    if t21 != 1 or t22 != 1:
        T2 = T2in.copy()
        T2 = np.triu(T2) + np.triu(T2, 1).T
    else:
        T2 = T2in[0, 0] * np.ones((HD, HD))
    T1[T1 == 0] = 1e10
    T2[T2 == 0] = 1e10
    Gamma = np.zeros((HD ** 2, HD ** 2))
    kk, jj = 1, 2
    for k1 in range(1, HD ** 2 + 1, HD + 1):                    # 1-based, as in MATLAB
        for k2 in range(k1 + HD + 1, HD ** 2 + 1, HD + 1):
            Gamma[k1 - 1, k2 - 1] = -1 / T1[kk - 1, jj - 1]
            if T1ud:
                Gamma[k2 - 1, k1 - 1] = -1 / T1[jj - 1, kk - 1]
            else:
                Gamma[k2 - 1, k1 - 1] = -1 / T1[kk - 1, jj - 1]
            jj += 1
        kk += 1
        jj = kk + 1
    n = 1
    T2vec = vec(T2)
    for k in range(2, HD ** 2):
        if k != n + 1 + n * HD:
            Gamma[k - 1, k - 1] = 1 / T2vec[k - 1]
        else:
            n += 1
    return Gamma
