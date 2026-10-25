"""Nelder/Mead simplex minimization algorithm"""
import time

import numpy as np

from .._registry import F, nout
from .._struct import Struct
from .._mat import isempty


def _call(fcn, x):
    """MATLAB [F,UserStop] = fcn(x) for functions returning a value or a tuple."""
    r = fcn(x)
    if isinstance(r, tuple):
        return float(np.real(r[0])), (bool(r[1]) if len(r) > 1 else False)
    return float(np.real(r)), False


def _rand(rng, n):
    return (np.random if rng is None else rng).random(n)


@nout(2)
def esfit_simplex(fcn=None, x0=None, lb=None, ub=None, FitOpt=None, rng=None):
    """esfit_simplex(fcn, x0, lb, ub, FitOpt) -> (x, info). fcn(x) returns the error value
    (or a tuple (value, userstop)). x is 1-D. FitOpt.RandomStart draws from rng (numpy
    Generator/RandomState; default the global numpy stream, so np.random.seed(s) reproduces
    MATLAB rng(s); rand)."""
    if fcn is None:
        return None
    if x0 is None or lb is None or ub is None:
        raise ValueError('At least 4 inputs expected (function, x0, lb, ub).')
    if FitOpt is None:
        FitOpt = Struct()
    DefOpt, _ = F.esfit_algdefaults('Nelder-Mead simplex')
    FitOpt = F.adddefaults(FitOpt, DefOpt)
    x0 = np.asarray(x0, dtype=float).reshape(-1, order='F')
    if not isinstance(FitOpt.get('SimplexPars'), (list, tuple, np.ndarray)) or isempty(FitOpt.SimplexPars):
        FitOpt.SimplexPars = [1, 2, 0.5, 0.5]
        n = x0.size
        if n > 2:
            FitOpt.SimplexPars = [1, 1 + 2 / n, 0.75 - 1 / (2 * n), 1 - 1 / n]
    rho, chi, psi, sigma = (float(v) for v in np.asarray(FitOpt.SimplexPars, dtype=float).reshape(-1)[:4])
    delta = FitOpt.delta
    lb = np.asarray(lb, dtype=float).reshape(-1, order='F')
    ub = np.asarray(ub, dtype=float).reshape(-1, order='F')
    if lb.size != x0.size:
        raise ValueError('Lower-bounds array must have the same number of elements as x0.')
    if ub.size != x0.size:
        raise ValueError('Upper-bounds array must have the same number of elements as x0.')
    if np.any(lb > ub):
        raise ValueError('Lower bounds must not be greater than upper bounds.')
    nParams = x0.size
    if np.any(np.isnan(x0)) or np.any(np.isinf(x0)):
        raise ValueError('x0 must be real and finite.')
    if np.any(x0 < lb) or np.any(x0 > ub):
        raise ValueError('Some elements in x0 are out of bounds.')
    if FitOpt.RandomStart:
        x0 = lb + _rand(rng, nParams) * (ub - lb)
    transformParams = bool(FitOpt.ScaleParams)
    if transformParams:
        lb0, ub0 = lb, ub
        transform = lambda x: 2 * (x - lb0) / (ub0 - lb0) - 1
        untransform = lambda x: lb0 + (ub0 - lb0) * (x / 2 + 1 / 2)
        x0 = transform(x0)
        ub = transform(ub0)
        lb = transform(lb0)
        fcn0 = fcn
        fcn = lambda x: fcn0(untransform(x))
    constrain = lambda x: np.maximum(np.minimum(x, ub), lb)

    iIteration = 0
    nEvals = 0
    stopCode = 0
    startTime = time.process_time()
    elapsedTime = 0.0
    nVertices = nParams + 1
    v = np.repeat(x0[:, None], nVertices, axis=1)
    for k in range(nParams):                       # v(nParams+1:nParams+1:end) += delta*(ub-lb).'
        v[k, k + 1] += delta * (ub[k] - lb[k])
    info = Struct()
    info.simplex_initial = v.copy()
    fv = np.zeros(nVertices)
    for iVertex in range(nVertices - 1, -1, -1):
        x = constrain(v[:, iVertex])
        fv[iVertex], UserStop = _call(fcn, x)
        nEvals += 1
        if UserStop:
            info.stop = 2
            return None, info
    idx = np.argsort(fv, kind='stable')
    fv = fv[idx]
    v = v[:, idx]
    Procedure = 'initial simplex'
    iIteration += 1
    logStringFormat = ' iteration %3d: value %0.5e   edge %0.5e   %s'
    if FitOpt.Verbosity > 0:
        FitOpt.IterationPrintFunction(logStringFormat % (iIteration, fv[0], delta, Procedure))

    while True:
        fv_old = fv.copy()
        elapsedTime = (time.process_time() - startTime) / 60
        if elapsedTime > FitOpt.maxTime:
            stopCode = 1
            break
        largestValDiff = np.max(np.abs(fv[0] - fv[1:nParams + 1]))
        longestEdgeLength = np.max(np.abs(v[:, 1:nParams + 1] - v[:, [0]]))
        if largestValDiff <= FitOpt.TolFun and longestEdgeLength <= FitOpt.TolEdgeLength:
            stopCode = 3
            break
        xbar = np.mean(v[:, :-1], axis=1)
        xr = xbar + rho * (xbar - v[:, -1])
        xr = constrain(xr)
        fr, _ = _call(fcn, xr)
        nEvals += 1
        doReduction = False
        if fr < fv[0]:
            xe = xbar + rho * chi * (xbar - v[:, -1])
            xe = constrain(xe)
            fe, _ = _call(fcn, xe)
            nEvals += 1
            if fe < fr:
                Procedure = 'expansion'
                v[:, -1] = xe
                fv[-1] = fe
            else:
                Procedure = 'reflection'
                v[:, -1] = xr
                fv[-1] = fr
        elif fr < fv[nParams - 1]:
            Procedure = 'reflection'
            v[:, -1] = xr
            fv[-1] = fr
        elif fr < fv[-1]:
            xco = xbar + psi * rho * (xbar - v[:, -1])
            xco = constrain(xco)
            fco, _ = _call(fcn, xco)
            nEvals += 1
            if fco <= fr:
                Procedure = 'contraction outside'
                v[:, -1] = xco
                fv[-1] = fco
            else:
                doReduction = True
        else:
            xci = xbar - psi * (xbar - v[:, -1])
            xci = constrain(xci)
            fci, _ = _call(fcn, xci)
            nEvals += 1
            if fci < fv[-1]:
                Procedure = 'contraction inside'
                v[:, -1] = xci
                fv[-1] = fci
            else:
                doReduction = True
        if doReduction:
            Procedure = 'reduction'
            for iVertex in range(1, nVertices):
                xshr = v[:, 0] + sigma * (v[:, iVertex] - v[:, 0])
                xshr = constrain(xshr)
                fv[iVertex], UserStop = _call(fcn, xshr)
                nEvals += 1
                v[:, iVertex] = xshr
                if UserStop:
                    stopCode = 2
                    break
        idx = np.argsort(fv, kind='stable')
        fv = fv[idx]
        v = v[:, idx]
        iIteration += 1
        info.bestx = v[:, 0].copy()
        info.minF = fv[0]
        info.nEvals = nEvals
        info.iter = iIteration
        info.newbest = bool(fv[0] < fv_old[0])
        UserStop = bool(FitOpt.IterFcn(info)) if not isempty(FitOpt.IterFcn) else False
        if UserStop:
            stopCode = 2
        if FitOpt.Verbosity > 0:
            thisstep = np.max(np.abs(v[:, 1:nParams + 1] - v[:, [0] * nParams]))
            FitOpt.IterationPrintFunction(logStringFormat % (iIteration, fv[0], thisstep, Procedure))
        if stopCode != 0:
            break
    x = untransform(v[:, 0]) if transformParams else v[:, 0].copy()
    info.simplex_final = v
    info.F = fv[0]
    info.nIterations = iIteration
    info.elapsedTime = elapsedTime
    if FitOpt.Verbosity > 0:
        msg = {1: 'Time limit of %f minutes reached.' % FitOpt.maxTime,
               2: 'Stopped by user.',
               3: 'Converged (edge length < %g and all errors < %g).' % (FitOpt.TolEdgeLength, FitOpt.TolFun)
               }.get(stopCode, '')
        FitOpt.InfoPrintFunction('Terminated: %s\n' % msg)
    info.stop = stopCode
    return x, info
