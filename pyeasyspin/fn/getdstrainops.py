"""Hamiltonian derivatives with respect to D and E, taking into account D/E strain correlation."""
import numpy as np

from .._registry import F, nout
from .._mat import vec
from .resfreqs_matrix import _d


@nout(3)
def getdstrainops(Sys=None):
    """(useDStrain, dHdD, dHdE): dHdD/dHdE are lists (MATLAB cells) with one operator per electron
    spin, already scaled by the strain widths (or by the principal axes of the D/E covariance)."""
    nEl = int(Sys.nElectrons)
    DStrain = np.atleast_2d(np.asarray(Sys.DStrain, dtype=float))
    useDStrain = bool(np.any(DStrain != 0))
    dHdD = [np.zeros((0, 0)) for _ in range(nEl)]
    dHdE = [np.zeros((0, 0)) for _ in range(nEl)]
    if not useDStrain:
        return useDStrain, dHdD, dHdE
    DFrame = np.atleast_2d(np.asarray(Sys.DFrame, dtype=float))
    corr = vec(np.asarray(Sys.DStrainCorr, dtype=float))
    for iEl in range(nEl):
        if not np.any(DStrain[iEl, 0:2]):
            dHdD[iEl] = 0.0
            dHdE[iEl] = 0.0
            continue
        Sx, Sy, Sz = (_d(F.sop(Sys, np.array([[iEl + 1, float(k)]]))) for k in (1, 2, 3))
        if np.any(DFrame[iEl, :]):
            R = F.erot(DFrame[iEl, :])
            SxD = R[0, 0] * Sx + R[0, 1] * Sy + R[0, 2] * Sz
            SyD = R[1, 0] * Sx + R[1, 1] * Sy + R[1, 2] * Sz
            SzD = R[2, 0] * Sx + R[2, 1] * Sy + R[2, 2] * Sz
            SxD2, SyD2, SzD2 = SxD @ SxD, SyD @ SyD, SzD @ SzD
        else:
            SxD2, SyD2, SzD2 = Sx @ Sx, Sy @ Sy, Sz @ Sz
        dHdD_ = (2 * SzD2 - SxD2 - SyD2) / 3
        dHdE_ = SxD2 - SyD2
        DeltaD, DeltaE = DStrain[iEl, 0], DStrain[iEl, 1]
        rDE = corr[iEl]
        if rDE != 0:
            F.logmsg(1, '  correlated D strain for electron spin %d (r = %f)', iEl + 1, rDE)
            R12 = rDE * DeltaD * DeltaE
            L, V = np.linalg.eigh(np.array([[DeltaD ** 2, R12], [R12, DeltaE ** 2]]))
            L = np.lib.scimath.sqrt(L)
            dHdD[iEl] = L[0] * (V[0, 0] * dHdD_ + V[0, 1] * dHdE_)
            dHdE[iEl] = L[1] * (V[1, 0] * dHdD_ + V[1, 1] * dHdE_)
        else:
            dHdD[iEl] = DeltaD * dHdD_
            dHdE[iEl] = DeltaE * dHdE_
    return useDStrain, dHdD, dHdE
