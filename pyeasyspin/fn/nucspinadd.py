"""Add a nuclear spin to a spin system."""
import numpy as np
from .._registry import F
from .._struct import Struct
from .._mat import a2, vec, isempty, numel, size


def _setrow(M, i, row):
    """MATLAB M(i,:) = row with zero-padding growth (i 1-based)."""
    M = a2(M)
    row = vec(row)
    if M.size == 0:
        M = np.zeros((0, row.size))
    r, c = M.shape
    out = np.zeros((max(r, i), max(c, row.size)), dtype=np.result_type(M, row))
    out[:r, :c] = M
    out[i - 1, :] = row if row.size == out.shape[1] else np.resize(row, out.shape[1])
    return out


def _setlin(M, i, v):
    """MATLAB M(i) = v for a vector M with growth (rows stay rows, columns stay columns)."""
    M = a2(M)
    if M.shape[0] > 1 and M.shape[1] == 1:
        out = np.zeros((max(M.shape[0], i), 1)); out[:M.shape[0], 0] = M[:, 0]; out[i - 1, 0] = v
    else:
        f = vec(M)
        out = np.zeros((1, max(f.size, i))); out[0, :f.size] = f; out[0, i - 1] = v
    return out


def _fullifyA(A, AFrame):
    A = vec(A)
    if A.size == 1:
        A = A[[0, 0, 0]]
    elif A.size == 2:
        A = A[[0, 0, 1]]
    elif A.size != 3:
        raise ValueError('A must contain 1, 2, or 3 elements.')
    if isempty(AFrame) or np.all(vec(AFrame) == 0):
        return np.diag(A)
    R = np.asarray(F.erot(AFrame))
    return R @ np.diag(A) @ R.T


def _fullifyQ(Q, QFrame, I):
    Q = vec(Q)
    I = vec(I)
    if I.size != 1:
        raise ValueError('Inner matrix dimensions must agree.')
    I = I[0]
    if Q.size in (1, 2):
        eta = Q[1] if Q.size == 2 else 0
        Qpv = Q[0] / (4 * I * (2 * I - 1)) * np.array([-1 + eta, -1 - eta, 2])
    elif Q.size == 3:
        Qpv = Q
    else:
        raise ValueError('Q must contain 1, 2, or 3 elements.')
    if isempty(QFrame) or np.all(vec(QFrame) == 0):
        return np.diag(Qpv)
    R = np.asarray(F.erot(QFrame))
    return R @ np.diag(Qpv) @ R.T


def _append(T0, T0Frame, T, TFrame, nNuclei, AQ, I=None):
    isA = AQ == 'A'
    if not isA and isempty(T):
        T = np.zeros((1, 3))
    T0 = a2(T0)
    T = a2(T)
    if T0.shape[0] == 3 * nNuclei:
        nT0 = 9
    elif T0.size == nNuclei:
        nT0 = 1
    elif T0.shape[1] == 2:
        nT0 = 2
    elif T0.shape[1] == 3:
        nT0 = 3
    fullT0 = nT0 == 9
    nT = T.size
    fullT = nT == 9
    if isA:
        one2two = lambda X: X[:, [0, 0]]
        one2three = lambda X: X[:, [0, 0, 0]]
        two2three = lambda X: X[:, [0, 0, 1]]
    else:
        Ir = vec(I).reshape(1, -1)
        one2two = lambda X: np.hstack([vec(X).reshape(-1, 1), np.zeros((X.size, 1))])
        one2three = lambda X: (vec(X).reshape(-1, 1) / (4 * Ir * (2 * Ir - 1))) @ np.array([[-1, -1, 2]])
        two2three = lambda X: (X[:, [0]] / (4 * Ir * (2 * Ir - 1))) @ np.array([[1, 1, 1]]) * 0 + \
            (X[:, [0]] / (4 * Ir * (2 * Ir - 1))) * np.hstack([-1 + X[:, [1]], -1 - X[:, [1]], 2 + 0 * X[:, [1]]])
    Tnew = T0
    new = nNuclei + 1
    if fullT0 or fullT:
        if not fullT0:
            tmp = np.zeros((3 * nNuclei, 3))
            Fr = np.zeros((nNuclei, 3)) if isempty(T0Frame) else a2(T0Frame)
            for i in range(nNuclei):
                tmp[3 * i:3 * i + 3, :] = (_fullifyA(T0[i, :], Fr[i, :]) if isA
                                           else _fullifyQ(T0[i, :], Fr[i, :], vec(I)[i]))
            Tnew = np.vstack([tmp, T])
        elif not fullT:
            Tnew = np.vstack([T0, _fullifyA(T, TFrame) if isA else _fullifyQ(T, TFrame, I)])
        else:
            Tnew = np.vstack([T0, T])
    else:
        if nT == 1:
            if nT0 == 1:
                Tnew = _setlin(Tnew, new, T.item())
            elif nT0 == 2:
                Tnew = _setrow(Tnew, new, one2two(T))
            else:
                Tnew = _setrow(Tnew, new, one2three(T))
        elif nT == 2:
            if nT0 == 1:
                Tnew = np.vstack([one2two(vec(Tnew).reshape(-1, 1)), T])
            elif nT0 == 2:
                Tnew = _setrow(Tnew, new, T)
            else:
                Tnew = _setrow(Tnew, new, two2three(T))
        else:
            if nT0 == 1:
                Tnew = np.vstack([one2three(vec(Tnew).reshape(-1, 1)), T])
            elif nT0 == 2:
                Tnew = np.vstack([two2three(Tnew), T])
            else:
                Tnew = np.vstack([Tnew, T])
    return Tnew


def _clean(Sys):
    New = Struct(Sys)
    for f in ('AFrame', 'Q', 'QFrame'):
        if f in Sys and (isempty(Sys[f]) or np.all(vec(Sys[f]) == 0)):
            del New[f]
    return New


def nucspinadd(Sys=None, Nuc=None, A=None, AFrame=None, Q=None, QFrame=None):
    """Spin system with nucleus Nuc ('14N') added, with hyperfine A (1, 2, 3 or 9 values),
    optional AFrame, quadrupole Q and QFrame (single-electron systems only)."""
    if Nuc is None:
        raise ValueError('Second input (isotope) is required.')
    if A is None:
        raise ValueError('Third input (hyperfine tensor) is required.')
    if not isinstance(Sys, dict):
        raise ValueError('First input argument must be a spin system structure!')
    if 'S' in Sys and numel(Sys['S']) > 1:
        raise ValueError('nucspinadd does not work if the system contains more than one electron spin.')
    if 'nn' in Sys and not isempty(Sys['nn']) and np.any(vec(Sys['nn']) != 0):
        raise ValueError('nucspinadd does not work if Sys.nn is present.')
    if not isinstance(Nuc, str):
        raise ValueError("Second input (Nuc) must be a character array, such as '14N'.")
    if isempty(AFrame):
        AFrame = np.array([[0.0, 0.0, 0.0]])
    if isempty(QFrame):
        QFrame = np.array([[0.0, 0.0, 0.0]])
    if numel(A) not in (1, 2, 3, 9):
        raise ValueError('Wrong size of hyperfine tensor (3rd input argument).')
    if numel(AFrame) != 3:
        raise ValueError('Wrong size of AFrame (4th input argument).')
    if not isempty(Q) and numel(Q) not in (1, 2, 3, 9):
        raise ValueError('Wrong size of quadrupole tensor (5th input argument).')
    if numel(QFrame) != 3:
        raise ValueError('Wrong size of QFrame (6th input argument).')
    if 'Nucs' in Sys:
        Nucs = F.nucstring2list(Sys['Nucs'])
        nNuclei = len(Nucs)
    else:
        nNuclei = 0
    New = Struct(Sys)
    if 'AFrame' not in New:
        New.AFrame = np.zeros((0, 0))
    if 'Q' not in New:
        New.Q = np.zeros((nNuclei, 3))
    if 'QFrame' not in New:
        New.QFrame = np.zeros((0, 0))
    iNuc = nNuclei + 1
    if nNuclei == 0:
        New.Nucs, New.A, New.AFrame = Nuc, A, AFrame
        New.Q = Q if Q is not None else np.zeros((0, 0))
        New.QFrame = QFrame
        return _clean(New)
    nucs = New.Nucs if isinstance(New.Nucs, str) else ','.join(New.Nucs)
    New.Nucs = nucs + ',' + Nuc
    if 'n' in New:
        New.n = _setlin(New.n, iNuc, 1)
    New.A = _append(New.A, New.AFrame, A, AFrame, nNuclei, 'A')
    fullA = numel(A) == 9 or size(a2(Sys['A']), 1) == 3 * nNuclei
    if not fullA:
        New.AFrame = _setrow(New.AFrame, iNuc, AFrame)
    if 'Q' in Sys or (Q is not None and np.any(vec(Q) != 0)):
        I = F.nucspin(Nucs)
        New.Q = _append(New.Q, New.QFrame, Q, QFrame, nNuclei, 'Q', I)
        fullQ = numel(Q) == 9 or size(a2(New.Q), 1) == 3 * nNuclei
        if not fullQ:
            New.QFrame = _setrow(New.QFrame, iNuc, QFrame)
    return _clean(New)
