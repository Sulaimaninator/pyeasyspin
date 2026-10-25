"""Orientation selection: weights of orientations excited at a given field and frequency."""
import itertools

import numpy as np

from .._registry import F, nout
from .._struct import Struct
from .._mat import vec, isempty, numel


@nout(-1)
def orisel(Sys=None, Params=None, Options=None, nargout=1, plot=False):
    """nargout 1: summed weights per orientation (column); nargout 2: (Weights, Transitions).
    plot=True draws the selection (MATLAB plots when called without outputs)."""
    if Options is None:
        Options = Struct()
    Sys, err = F.validatespinsys(Sys)
    if err:
        raise ValueError(err)
    if 'HStrain' not in Sys:
        Sys.HStrain = np.array([[0.0, 0.0, 0.0]])
    if numel(Sys.HStrain) == 1:
        Sys.HStrain = np.ones((1, 3)) * float(np.asarray(Sys.HStrain).ravel()[0])
    for f in ('Field', 'mwFreq', 'ExciteWidth'):
        if f not in Params:
            raise ValueError(f'Params.{f} is missing!')
    Params = Struct(F.adddefaults(Params, Struct(mwFreq=np.nan, Field=np.nan, ExciteWidth=np.inf,
                                                 Orientations=np.zeros((0, 0)))))
    HStrain = vec(np.asarray(Sys.HStrain, dtype=float))[:3]
    ExciteWidth = float(np.asarray(Params.ExciteWidth).ravel()[0])
    if np.any(HStrain + ExciteWidth <= 0):
        raise ValueError('Sys.HStrain or Exp.ExciteWidth must be positive!')
    mwFreq = float(np.asarray(Params.mwFreq).ravel()[0]) * 1e3
    Field = float(np.asarray(Params.Field).ravel()[0])
    Options = Struct(F.adddefaults(Options, Struct(GridSize=46.0, GridSymmetry='', Display=nargout == 0,
                                                   AveragedIntensity=True)))
    if not isempty(Params.Orientations):
        Orientations = np.atleast_2d(np.asarray(Params.Orientations, dtype=float))
        phi, theta = Orientations[0, :], Orientations[1, :]
    else:
        if isempty(Options.GridSymmetry):
            Options.GridSymmetry, GridFrame = F.hamsymm(Sys)[:2]
        else:
            GridFrame = np.eye(3)
        gridopt = 'c' if bool(Options.Display) else ''
        grid, tri = F.sphgrid(Options.GridSymmetry, Options.GridSize, gridopt, nargout=2)[:2]
        phi, theta = (vec(np.asarray(x, dtype=float)) for x in F.vec2ang(np.asarray(GridFrame) @ np.asarray(grid.vecs))[:2])
        Orientations = np.vstack([phi, theta])
    nAngles, nOrientations = Orientations.shape
    if nAngles == 2:
        Orientations = np.vstack([Orientations, np.zeros((1, nOrientations))])
    elif nAngles != 3:
        raise ValueError('Orientations array has wrong size, should be 2xn or 3xn.')
    H0, muxM, muyM, muzM = (np.asarray(x.toarray() if hasattr(x, 'toarray') else x) for x in F.ham(Sys, nargout=4)[:4])
    N = H0.shape[0]
    Transitions = np.array(list(itertools.combinations(range(1, N + 1), 2)), dtype=float).reshape(-1, 2)
    u, v = Transitions[:, 0].astype(int), Transitions[:, 1].astype(int)
    uv = u + (v - 1) * N
    fac = 1 / np.sqrt(2 * np.log(2))
    if np.isfinite(ExciteWidth):
        Weights = np.zeros((nOrientations, u.size))
        vec_ = np.asarray(F.ang2vec(Orientations[0, :], Orientations[1, :]), dtype=float).T
        GammaWidth = fac * np.sqrt(np.sum(vec_ ** 2 * HStrain ** 2, axis=1) + ExciteWidth ** 2)
        for iOri in range(nOrientations):
            xLab, yLab, zLab = (vec(np.asarray(x, dtype=float)) for x in
                                F.erot(phi[iOri], theta[iOri], 0.0, 'rows', nargout=3))
            muxL = xLab[0] * muxM + xLab[1] * muyM + xLab[2] * muzM
            muyL = yLab[0] * muxM + yLab[1] * muyM + yLab[2] * muzM
            muzL = zLab[0] * muxM + zLab[1] * muyM + zLab[2] * muzM
            E, V = np.linalg.eigh(H0 - Field * muzL)
            idx = np.argsort(E, kind='stable')
            E, V = E[idx], V[idx, :]           # EasySpin reorders rows (idx is the identity here)
            if bool(Options.AveragedIntensity):
                TR = (np.abs(V.conj().T @ muxL @ V) ** 2 + np.abs(V.conj().T @ muyL @ V) ** 2) / 2
            else:
                TR = np.abs(V.conj().T @ muxL @ V) ** 2
            xi = (np.abs(E[v - 1] - E[u - 1]) - mwFreq) / GammaWidth[iOri]
            Weights[iOri, :] = TR.ravel(order='F')[uv - 1] * np.exp(-2 * xi ** 2)
    else:
        Weights = np.ones((nOrientations, u.size))
    if plot:
        import matplotlib.pyplot as plt
        w = Weights.sum(axis=1)
        if isempty(Params.Orientations) and Options.GridSymmetry == 'Dinfh':
            plt.plot(theta * 180 / np.pi, w)
            plt.xlabel('theta [deg]')
            plt.ylabel('weights [a.u.]')
        else:
            plt.scatter(phi * 180 / np.pi, theta * 180 / np.pi, c=w)
            plt.xlabel('phi [deg]')
            plt.ylabel('theta [deg]')
            plt.colorbar()
    if nargout == 2:
        return Weights, Transitions
    return Weights.sum(axis=1).reshape(-1, 1)
