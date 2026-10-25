"""Computes linewidth parameters for fast-motion regime"""
import numpy as np
from .._registry import F, nout
from .._struct import Struct, isfield
from .._mat import a2, vec, size


def _c(name):
    v = getattr(F, name)()
    return v[0] if isinstance(v, tuple) else v


def _rot_diag(R, d):
    """MATLAB R*diag(d)*R.' (diag of a scalar is the scalar itself)."""
    d = vec(np.asarray(d, dtype=float))
    if d.size == 1:
        return d[0] * (R @ R.T)
    if d.size != 3:
        raise ValueError('Inner matrix dimensions must agree.')
    return R @ np.diag(d) @ R.T


def _convertcoeffs(c_angfrq):
    c = c_angfrq / 1e6 / 1
    c = c / np.pi
    return c


def _all_mI(I):
    I = vec(np.asarray(I, dtype=float))
    mIall = np.zeros((1, 0))
    nI = (2 * I + 1).round().astype(int)
    n = 1
    for iNuc in range(I.size - 1, -1, -1):
        m = np.arange(-I[iNuc], I[iNuc] + 0.5)          # -I:I
        col1 = np.repeat(m, n)                           # repmat(m, n, 1)(:)
        n = n * nI[iNuc]
        rest = np.tile(mIall, (nI[iNuc], 1)) if mIall.shape[1] else np.zeros((col1.size, 0))
        mIall = np.column_stack([col1, rest])
    return mIall


@nout(3)
def fastmotion(System=None, Field=None, tau20=None, domain=None):
    """fastmotion(Sys, Field (mT), tau20 (s), domain='field'|'freq') -> (lw, mI, coeffs).
    lw: Lorentzian FWHM per hyperfine line (column; mT for 'field', MHz for 'freq'), mI: the
    mI combinations of the lines (rows), coeffs: Struct of the A, B, C, D, qA, qC, qE coefficients.
    Without nuclei, lw is the scalar A and mI is empty (MATLAB leaves it unassigned)."""
    if domain is None:
        domain = 'field'
    if domain not in ('freq', 'field'):
        raise ValueError("Fourth input must be either 'field' or 'freq'.")
    fieldUnits = domain == 'field'
    System = Struct(System)
    if isfield(System, 'Nucs'):
        I = vec(np.asarray(F.nucspin(System.Nucs), dtype=float))
        nNucs = I.size
    else:
        I = np.zeros(0)
        nNucs = 0
    if isfield(System, 'S'):
        S = vec(np.asarray(System.S, dtype=float))
        if S.size != 1 or S[0] != 0.5:
            raise ValueError('  Fast-motion linewidths can only be calculated for systems with S=1/2.')
    gS = a2(System.g).astype(float)
    anisotropic_g = np.max(gS) != np.min(gS)
    anisotropic_A = False
    if nNucs > 0:
        Asys = a2(System.A).astype(float)
        nr = Asys.shape[0]
        if nr == nNucs:
            anisotropic_A = bool(np.any(np.max(Asys, axis=1) != np.min(Asys, axis=1)))
        elif nr == 3 * nNucs:
            anisotropic_A = False
            for iNuc in range(nNucs):
                diagA = np.diag(Asys[iNuc * 3, :])     # MATLAB diag of a row: a 3x3 matrix
                v = np.max(diagA, axis=0) != np.min(diagA, axis=0)
                if not anisotropic_A:
                    if v.size != 1:
                        raise ValueError('Operands to the || and && operators must be convertible '
                                         'to logical scalar values.')
                    anisotropic_A = bool(v)
        else:
            raise ValueError('System.A has incorrect size.')
    if tau20 < 1e-14:
        raise ValueError('Correlation time too small!')
    if not anisotropic_g and not anisotropic_A and not isfield(System, 'Q'):
        raise ValueError('Either g or A must be anisotropic, or Q must be present.')

    fullg = gS.size == 9
    if fullg:
        g = gS
    else:
        if isfield(System, 'gFrame'):
            R_g2M = np.asarray(F.erot(System.gFrame)).T
        else:
            R_g2M = np.eye(3)
        g = _rot_diag(R_g2M, gS)
    g0 = np.trace(g) / 3
    g1 = vec(g - np.eye(3) * g0).reshape(9, 1)

    if not isfield(System, 'AFrame'):
        System.AFrame = np.zeros((nNucs, 3))
    if not isfield(System, 'Q'):
        System.Q = np.zeros((nNucs, 3))
    if not isfield(System, 'QFrame'):
        System.QFrame = np.zeros((nNucs, 3))
    A0 = np.zeros(nNucs)
    A1 = np.zeros((9, nNucs))
    if nNucs > 0:
        AFrame = a2(System.AFrame).astype(float)
        for iNuc in range(nNucs - 1, -1, -1):
            R_A2M = np.asarray(F.erot(AFrame[iNuc, :])).T
            diagA = Asys[iNuc, :]
            if diagA.size == 2:
                diagA = diagA[[0, 0, 1]]
            A = _rot_diag(R_A2M, diagA)
            A0[iNuc] = np.trace(A) / 3
            A1[:, iNuc] = vec(A - np.eye(3) * A0[iNuc])
        A1 = A1 * 1e6 * (2 * np.pi)

    cc = _c('bmagn') * Field * 1e-3 / (_c('planck') / 2 / np.pi)
    omega0 = g0 * cc
    j0 = tau20
    j1 = tau20 / (1 + omega0 ** 2 * tau20 ** 2)

    if nNucs > 0:
        AA = A1.T @ A1
        gA = cc * (g1.T @ A1)                 # 1 x nNucs
        dAA = np.diag(AA).reshape(1, -1)
    else:
        AA = 0.0
        gA = 0.0
        dAA = np.zeros((1, 0))
    gg = cc ** 2 * float((g1.T @ g1).item())
    Q = a2(System.Q).astype(float)
    PP = (np.sum(Q ** 2, axis=1).reshape(1, -1) if Q.size else np.zeros((1, 0))) * (1e6 * (2 * np.pi)) ** 2

    A = gg * (2 / 15 * j0 + 1 / 10 * j1) + np.sum(I * (I + 1) * vec(dAA)) * (1 / 20 * j0 + 7 / 60 * j1)
    B = gA * (4 / 15 * j0 + 1 / 5 * j1)
    if nNucs > 0:
        C = dAA * (1 / 12 * j0 - 1 / 60 * j1)
        D = AA * (4 / 15 * j0 + 1 / 10 * j1)
        D = D - np.tril(D)
    else:
        C = 0.0 * (1 / 12 * j0 - 1 / 60 * j1)
        D = 0.0
    A = _convertcoeffs(A)
    B = _convertcoeffs(B)
    C = _convertcoeffs(C)
    D = _convertcoeffs(D)
    if fieldUnits:
        A = F.unitconvert(A, 'MHz->mT', g0)
        B = F.unitconvert(B, 'MHz->mT', g0)
        C = F.unitconvert(C, 'MHz->mT', g0)
        D = F.unitconvert(D, 'MHz->mT', g0)
    coeffs = Struct()
    coeffs.A = A
    coeffs.B = B
    coeffs.C = C
    coeffs.D = D

    if nNucs > 0:
        PPv = vec(PP)
        II = I * (I + 1)
        qA = np.sum(j0 / 20 * PPv * II * (II - 1))
        qC = j0 / 20 * PPv * I * (I + 1) * 2
        qE = j0 / 20 * PPv * (-3)
        qA = _convertcoeffs(qA)
        qC = _convertcoeffs(qC)
        qE = _convertcoeffs(qE)
        if fieldUnits:
            qA = F.unitconvert(qA, 'MHz->mT', g0)
            qC = F.unitconvert(qC, 'MHz->mT', g0)
            qE = F.unitconvert(qE, 'MHz->mT', g0)
    else:
        qA = 0
        qC = 0
        qE = 0
    coeffs.qA = qA
    coeffs.qC = qC
    coeffs.qE = qE

    if nNucs > 0:
        mI = _all_mI(I)
        Bv, Cv, qCv, qEv = (vec(np.asarray(x, dtype=float)) for x in (B, C, qC, qE))
        Dm = np.asarray(D, dtype=float).reshape(nNucs, nNucs)
        A_ = float(np.ravel(A)[0])
        qA_ = float(np.ravel(qA)[0])
        lw = np.zeros(mI.shape[0])
        for iLine in range(mI.shape[0] - 1, -1, -1):
            m = mI[iLine, :]
            lw[iLine] = A_ + np.sum(Bv * m) + np.sum(Cv * m ** 2) + np.sum(Dm * np.outer(m, m))
            lw[iLine] = lw[iLine] + qA_ + np.sum(qCv * m ** 2) + np.sum(qEv * m ** 4)
        lw = lw.reshape(-1, 1)
    else:
        lw = A
        mI = np.zeros((0, 0))
    return lw, mI, coeffs
