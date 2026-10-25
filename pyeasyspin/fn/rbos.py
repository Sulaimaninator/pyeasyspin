"""takes as input the ISTO components and coefficients for each"""
import numpy as np
import scipy.sparse as sp
from .._registry import F as _F, nout
from .._mat import vec


def _sp(x):
    return x if sp.issparse(x) else sp.csc_matrix(np.atleast_2d(np.asarray(x)))


def _add(acc, term):
    # MATLAB: sparse(0) + term (scalar expansion of the 1x1 zero)
    return term if acc is None else acc + term


def _fin(acc):
    return sp.csc_matrix((1, 1)) if acc is None else sp.csc_matrix(acc)


@nout(6)
def rbos(T=None, F=None, angles=None, isFieldDep=None):
    """Rotational basis operators for chili: (Q0G, Q1G, Q2G, Q0F, Q1F, Q2F), the rank-0/1/2 ISTO
    Liouvillians for field-dependent (G) and field-independent (F) terms, rotated by the Euler
    angles. Q1*/Q2* are lists (3x3 / 5x5 cells, column-major); Q1* are [] if all F1 are zero."""
    a = vec(np.asarray(angles, dtype=float))
    D1 = np.asarray(_F.wignerd(1, a[0], a[1], a[2]))
    D2 = np.asarray(_F.wignerd(2, a[0], a[1], a[2]))
    T0 = T['T0'] if isinstance(T['T0'], list) else [T['T0']]
    T1 = T['T1'] if isinstance(T['T1'], list) else [T['T1']]
    T2 = T['T2'] if isinstance(T['T2'], list) else [T['T2']]
    F0 = vec(np.asarray(F['F0']))
    nTerms = F0.size
    F1 = np.asarray(F['F1']).reshape(nTerms, -1, order='F') if np.size(F['F1']) else np.zeros((nTerms, 0))
    F2 = np.asarray(F['F2']).reshape(nTerms, -1, order='F')
    fd = vec(np.asarray(isFieldDep)).astype(bool)

    def Tc(cell, iTerm, m):            # cell{iTerm,m} of an nTerms x K cell (column-major list)
        return _sp(cell[iTerm + m * nTerms])

    Q0G = Q0F = None
    for iTerm in range(nTerms):
        term = np.conj(F0[iTerm]) * _sp(T0[iTerm])
        if fd[iTerm]:
            Q0G = _add(Q0G, term)
        else:
            Q0F = _add(Q0F, term)
    Q0G, Q0F = _fin(Q0G), _fin(Q0F)

    anyF1 = bool(np.any(F1 != 0))
    if anyF1:
        Q1G = [None] * 9
        Q1F = [None] * 9
        for mp in range(3):
            for mq in range(3):
                QB = QG = None
                for m in range(3):
                    for iTerm in range(nTerms):
                        coef = D1[m, mp] * (-1) * np.conj(F1[iTerm, mq])
                        if fd[iTerm]:
                            QB = _add(QB, coef * Tc(T1, iTerm, m))
                        else:
                            QG = _add(QG, coef * Tc(T1, iTerm, m))
                Q1G[mp + 3 * mq] = _fin(QB)
                Q1F[mp + 3 * mq] = _fin(QG)
    else:
        Q1G = []
        Q1F = []

    Q2G = [None] * 25
    Q2F = [None] * 25
    for mp in range(5):
        for mq in range(5):
            QB = QG = None
            for m in range(5):
                for iTerm in range(nTerms):
                    coef = D2[m, mp] * np.conj(F2[iTerm, mq])
                    if fd[iTerm]:
                        QB = _add(QB, coef * Tc(T2, iTerm, m))
                    else:
                        QG = _add(QG, coef * Tc(T2, iTerm, m))
            Q2G[mp + 5 * mq] = _fin(QB)
            Q2F[mp + 5 * mq] = _fin(QG)

    if np.all(fd):
        nDim = max(_sp(T0[0]).shape)
        Q0F = sp.csc_matrix((nDim, nDim))
        if anyF1:
            Q1F = [sp.csc_matrix((nDim, nDim)) for _ in range(9)]
        Q2F = [sp.csc_matrix((nDim, nDim)) for _ in range(25)]

    Q0F = _F.hil2liouv(Q0F)
    Q0G = _F.hil2liouv(Q0G)
    Q1F = _F.hil2liouv(Q1F)
    Q1G = _F.hil2liouv(Q1G)
    Q2F = _F.hil2liouv(Q2F)
    Q2G = _F.hil2liouv(Q2G)
    return Q0G, Q1G, Q2G, Q0F, Q1F, Q2F
