"""Combination of equivalent spins."""
import numpy as np
from .._registry import nout


@nout(2)
def equivcouple(I=None, n=None):
    """(F, N): total spins F of n equivalent spins I and their multiplicities N."""
    if n == 1:
        return I, n
    Fv = np.arange(I * n, -1e-9, -1.0)
    E = np.ones(int(round(2 * I + 1)))
    RowVec = np.ones(1)
    for _ in range(int(n)):
        RowVec = np.convolve(RowVec, E)
    RowVec = np.concatenate([[0.0], RowVec])
    N = np.diff(RowVec)[:len(Fv)]
    return Fv, N
