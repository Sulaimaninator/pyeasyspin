"""Function minimization using grid search"""
import time

import numpy as np

from .._registry import F, nout
from .._struct import Struct
from .._mat import isempty


def _call(fcn, x):
    r = fcn(x)
    return float(np.real(r[0] if isinstance(r, tuple) else r))


@nout(2)
def esfit_grid(fcn=None, lb=None, ub=None, FitOpt=None, rng=None):
    """esfit_grid(fcn, lb, ub, FitOpt) -> (bestx, info). FitOpt.randGrid shuffles the grid with
    rng.permutation (numpy Generator/RandomState; default the global numpy stream)."""
    if fcn is None:
        return None
    if lb is None or ub is None:
        raise ValueError('At least 3 inputs expected (function, lb, ub).')
    if FitOpt is None:
        FitOpt = Struct()
    DefOpt, _ = F.esfit_algdefaults('grid search')
    FitOpt = F.adddefaults(FitOpt, DefOpt)
    lb = np.asarray(lb, dtype=float).reshape(-1, order='F')
    ub = np.asarray(ub, dtype=float).reshape(-1, order='F')
    if lb.size != ub.size:
        raise ValueError('Arrays for lower and upper bound must have the same number of elements.')
    if np.any(lb > ub):
        raise ValueError('Lower bounds must not be greater than upper bounds.')
    nParams = lb.size
    GridSize = np.asarray(FitOpt.GridSize).reshape(-1)
    if GridSize.size == 1:
        GridSize = GridSize[0] * np.ones(nParams)
    if GridSize.size != nParams:
        raise ValueError('FitOpt.GridSize must have as many elements as there are fitting parameters.')
    if np.any(GridSize < 2):
        raise ValueError('At least two grid points per parameter are needed.')
    GridSize = GridSize.astype(int)
    nGridPoints = int(np.prod(GridSize))
    if nGridPoints > FitOpt.maxGridPoints:
        raise ValueError('Cannot do grid search with more than %d points. Reduce number of parameters.'
                         % FitOpt.maxGridPoints)
    if FitOpt.Verbosity:
        FitOpt.InfoPrintFunction('%d parameters, %d grid points total\n' % (nParams, nGridPoints))
    if not isempty(FitOpt.IterFcn) and not callable(FitOpt.IterFcn):
        raise ValueError('Opt.IterFcn must be a function handle.')
    gridvals = [np.linspace(lb[p], ub[p], GridSize[p]) for p in range(nParams)]
    X = np.meshgrid(*gridvals, indexing='ij')                     # ndgrid
    X = np.stack([Xp.reshape(-1, order='F') for Xp in X], axis=0)  # nParams x nGridPoints
    if FitOpt.randGrid:
        X = X[:, (np.random if rng is None else rng).permutation(nGridPoints)]
    minF = np.inf
    bestx = np.full(nParams, np.nan)
    startTime = time.process_time()
    stopCode = 0
    nEvals = 0
    iIteration = 0
    info = Struct()
    for idx in range(nGridPoints):
        Fv = _call(fcn, X[:, idx].copy())
        nEvals += 1
        iIteration += 1
        newbest = Fv < minF
        if newbest:
            minF = Fv
            bestx = X[:, idx].copy()
            if FitOpt.Verbosity:
                FitOpt.IterationPrintFunction('  Point %4d/%d:   error %0.5e  best so far'
                                              % (iIteration, nGridPoints, Fv))
        info.currx = X[:, idx].copy()
        info.currF = Fv
        info.bestx = bestx
        info.minF = minF
        info.nEvals = nEvals
        info.iter = iIteration
        info.newbest = newbest
        UserStop = bool(FitOpt.IterFcn(info)) if not isempty(FitOpt.IterFcn) else False
        elapsedTime = (time.process_time() - startTime) / 60
        if elapsedTime > FitOpt.maxTime:
            stopCode = 1
        if UserStop:
            stopCode = 2
        if minF < FitOpt.TolFun:
            stopCode = 3
        if stopCode != 0:
            break
    if FitOpt.Verbosity > 0:
        msg = {0: 'Terminated: all grid points searched.',
               1: 'Terminated: Time limit of %f minutes reached.' % FitOpt.maxTime,
               2: 'Terminated: Stopped by user.',
               3: 'Terminated: Found a parameter set with error less than %g.' % FitOpt.TolFun}[stopCode]
        FitOpt.InfoPrintFunction(msg)
    info.stop = stopCode
    return bestx, info
