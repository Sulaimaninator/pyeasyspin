"""Transformation matrix between coupled and uncoupled representations"""
import numpy as np
from .._registry import F, nout
from .._mat import isempty


def _down(a, b):
    """MATLAB a:-1:b."""
    n = int(np.floor(a - b + 1e-9)) + 1
    return [a - i for i in range(max(n, 0))]


@nout(3)
def cgmatrix(S1=None, S2=None, Stot=None):
    """cgmatrix(S1, S2[, Stot]) -> (U2C, Smtot, m12)."""
    if S1 is None or S2 is None:
        raise ValueError('Wrong number of input parameters. cgmatrix needs two inputs: S1 and S2.')
    if np.size(S1) != 1 or np.size(S2) != 1:
        raise ValueError('S1 and S2 must be single numbers.')
    S1 = float(np.asarray(S1).ravel()[0])
    S2 = float(np.asarray(S2).ravel()[0])
    if (S1 * 2) % 1 or (S2 * 2) % 1 or S1 < 0.5 or S2 < 0.5:
        raise ValueError('S1 and S2 must be spin quantum numbers 1/2, 1, 3/2, etc.')
    if Stot is not None and not isempty(Stot):
        if np.size(Stot) != 1:
            raise ValueError('Invalid third input (Stot).')
        Stot = float(np.asarray(Stot).ravel()[0])
        if (Stot * 2) % 1 or Stot > S1 + S2 or Stot < abs(S1 - S2):
            raise ValueError('Invalid third input (Stot).')
        Stot = [Stot]
    else:
        Stot = _down(S1 + S2, abs(S1 - S2))
    nStates = int(round(sum(2 * s + 1 for s in Stot)))
    # MATLAB preallocates nStates but grows the arrays when only one Stot is requested
    nUnc = int(round((2 * S1 + 1) * (2 * S2 + 1)))
    U2C = np.zeros((nStates, max(nStates, nUnc)))
    Smtot = np.zeros((nStates, 2))
    m12 = np.zeros((max(nStates, nUnc), 2))
    iu = 0
    for mS1 in _down(S1, -S1):
        for mS2 in _down(S2, -S2):
            m12[iu, :] = [mS1, mS2]
            ic = 0
            for St in Stot:
                for mSt in _down(St, -St):
                    U2C[ic, iu] = np.real(F.clebschgordan(S1, S2, St, mS1, mS2, mSt))
                    if iu == 0:
                        Smtot[ic, :] = [St, mSt]
                    ic += 1
            iu += 1
    return U2C, Smtot, m12
