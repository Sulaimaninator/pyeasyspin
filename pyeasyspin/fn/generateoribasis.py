"""Orientational basis (L, jK, K, M quantum numbers) for slow-motion simulations."""
import numpy as np

from .._struct import Struct


def _basis(Basis, rows, names):
    B = Struct(Basis)
    arr = np.array(rows, dtype=float).reshape(-1, len(names))
    for j, n in enumerate(names):
        B[n] = arr[:, j:j + 1].copy()
    return B


def generateoribasis(Basis=None, basistype=None):
    """Basis with Llist and the quantum-number columns for 'LjKKM' (default), 'LMKjK' or 'LMK'."""
    if basistype is None:
        basistype = 'LjKKM'
    Basis = Struct(Basis)
    Leven = np.arange(0, float(Basis.evenLmax) + 1, 2)
    Lodd = np.arange(1, float(Basis.oddLmax) + 1, 2)
    Basis.Llist = np.sort(np.concatenate([Leven, Lodd])).reshape(1, -1)
    if 'jKmin' in Basis:
        if float(Basis.jKmin) not in (1.0, -1.0):
            raise ValueError('Basis.jKmin must be either -1 or +1.')
    else:
        Basis.jKmin = -1.0
    if 'evenK' not in Basis:
        Basis.evenK = False
    evenK = bool(np.all(np.asarray(Basis.evenK) != 0))
    if evenK:
        Basis.Kmax = float(np.floor(float(Basis.Kmax) / 2) * 2)
    Kmax, Mmax, jKmin = int(Basis.Kmax), int(Basis.Mmax), int(Basis.jKmin)
    deltaK = int(evenK) + 1
    rows = []
    if basistype == 'LjKKM':
        for L in Basis.Llist.ravel().astype(int):
            Lparity = 1 if L % 2 == 0 else -1
            for jK in range(jKmin, 2, 2):
                for K in range(0, min(L, Kmax) + 1, deltaK):
                    if K == 0 and Lparity != jK:
                        continue
                    Mmx = min(L, Mmax)
                    rows += [[L, jK, K, M] for M in range(-Mmx, Mmx + 1)]
        return _basis(Basis, rows, ('L', 'jK', 'K', 'M'))
    if basistype == 'LMKjK':
        for L in Basis.Llist.ravel().astype(int):
            Mmx = min(L, Mmax)
            for M in range(-Mmx, Mmx + 1):
                Kmx = min(L, Kmax)
                for K in range(-Kmx, Kmx + 1, deltaK):
                    jK = (-1) ** L if K == 0 else int(np.sign(K))
                    if jK < jKmin:
                        continue
                    rows.append([L, M, abs(K), jK])
        return _basis(Basis, rows, ('L', 'M', 'K', 'jK'))
    if basistype == 'LMK':
        for L in Basis.Llist.ravel().astype(int):
            Mmx = min(L, Mmax)
            for M in range(-Mmx, Mmx + 1):
                Kmx = min(L, Kmax)
                rows += [[L, M, K] for K in range(-Kmx, Kmx + 1, deltaK)]
        return _basis(Basis, rows, ('L', 'M', 'K'))
    return None
