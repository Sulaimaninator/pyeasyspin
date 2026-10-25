"""Monte Carlo algorithm for least-squares fitting"""
import time

import numpy as np

from .._registry import F, nout
from .._struct import Struct
from .._mat import isempty


def _call(fcn, x):
    r = fcn(x)
    return float(np.real(r[0] if isinstance(r, tuple) else r))


@nout(2)
def esfit_montecarlo(fcn=None, lb=None, ub=None, FitOpt=None, rng=None):
    """esfit_montecarlo(fcn, lb, ub, FitOpt) -> (bestx, info). Trial points
    lb + (ub-lb).*rand(n,1) are drawn from rng (numpy Generator/RandomState; default the global
    numpy stream: np.random.seed(s) reproduces MATLAB rng(s) draws exactly)."""
    if fcn is None:
        return None
    if lb is None or ub is None:
        raise ValueError('At least 3 inputs expected (function, lb, ub).')
    if FitOpt is None:
        FitOpt = Struct()
    DefOpt, _ = F.esfit_algdefaults('Monte Carlo')
    FitOpt = F.adddefaults(FitOpt, DefOpt)
    lb = np.asarray(lb, dtype=float).reshape(-1, order='F')
    ub = np.asarray(ub, dtype=float).reshape(-1, order='F')
    if lb.size != ub.size:
        raise ValueError('Arrays for lower and upper bound must have the same number of elements.')
    if np.any(lb > ub):
        raise ValueError('Lower bounds must not be greater than upper bounds.')
    src = np.random if rng is None else rng
    nParams = lb.size
    Fmin = np.inf
    bestx = np.zeros(nParams)
    startTime = time.process_time()
    stopCode = 0
    elapsedTime = 0.0
    info = Struct()
    iTrial = 0
    for iTrial in range(1, int(FitOpt.nTrials) + 1):
        X = lb + (ub - lb) * src.random(nParams)
        Fv = _call(fcn, X)
        newbest = Fv < Fmin
        if newbest:
            Fmin = Fv
            bestx = X
            if FitOpt.Verbosity:
                FitOpt.IterationPrintFunction('%d:   error %0.5e  best so far' % (iTrial, Fv))
        info.bestx = bestx
        info.minF = Fmin
        info.nEvals = iTrial
        info.iter = iTrial
        info.newbest = newbest
        UserStop = bool(FitOpt.IterFcn(info)) if not isempty(FitOpt.IterFcn) else False
        if UserStop:
            stopCode = 2
        elapsedTime = (time.process_time() - startTime) / 60
        if elapsedTime > FitOpt.maxTime:
            stopCode = 1
        if Fv < FitOpt.TolFun:
            stopCode = 3
        if stopCode:
            break
    if FitOpt.Verbosity > 0:
        msg = {0: 'Terminated: maximum number of trials reached.',
               1: 'Terminated: Time limit of %f minutes reached.' % FitOpt.maxTime,
               2: 'Terminated: Stopped by user.',
               3: 'Terminated: Found a parameter set with error less than %g.' % FitOpt.TolFun}[stopCode]
        FitOpt.InfoPrintFunction(msg)
    info.nTrials = iTrial
    info.elapsedTime = elapsedTime
    return bestx, info
