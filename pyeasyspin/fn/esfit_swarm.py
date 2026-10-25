"""Particle swarm optimization for least-squares fitting"""
import time

import numpy as np

from .._registry import F, nout
from .._struct import Struct
from .._mat import isempty


def _call(fcn, x):
    r = fcn(x)
    if isinstance(r, tuple):
        return float(np.real(r[0])), (bool(r[1]) if len(r) > 1 else False)
    return float(np.real(r)), False


def _rand(src, m, n):
    return src.random(m * n).reshape((m, n), order='F')


@nout(2)
def esfit_swarm(fcn=None, lb=None, ub=None, FitOpt=None, rng=None):
    """esfit_swarm(fcn, lb, ub, FitOpt) -> (gX, info). Random draws come from rng (numpy
    Generator/RandomState; default the global numpy stream: np.random.seed(s) reproduces
    MATLAB rng(s) exactly, since only rand is used)."""
    if fcn is None:
        return None
    if lb is None or ub is None:
        raise ValueError('At least 3 inputs expected (function, lb, ub).')
    if FitOpt is None:
        FitOpt = Struct()
    DefOpt, _ = F.esfit_algdefaults('particle swarm')
    FitOpt = F.adddefaults(FitOpt, DefOpt)
    src = np.random if rng is None else rng
    nParams = np.size(lb)
    if isempty(FitOpt.get('nParticles')):
        FitOpt.nParticles = 20 + nParams * 10
    lb = np.asarray(lb, dtype=float).reshape(-1, order='F')
    ub = np.asarray(ub, dtype=float).reshape(-1, order='F')
    if lb.size != ub.size:
        raise ValueError('Arrays for lower and upper bound must have the same number of elements.')
    if np.any(lb > ub):
        raise ValueError('Lower bounds must not be greater than upper bounds.')
    nParticles = int(FitOpt.nParticles)
    k, w, c1, c2 = (float(v) for v in np.asarray(FitOpt.SwarmParams, dtype=float).reshape(-1)[:4])
    if FitOpt.Verbosity:
        FitOpt.InfoPrintFunction(['Particle swarm optimization parameters:',
                                  '   n = %d (number of particles)' % nParticles,
                                  '   k = %g (velocity clamping)' % k,
                                  '   k = %g (inertial coefficient)' % w,
                                  '   c1 = %g (cognitive coefficient)' % c1,
                                  '   c2 = %g (social coefficient)' % c2])
        FitOpt.InfoPrintFunction('Initializing swarm...')
    X = lb[:, None] + (ub - lb)[:, None] * _rand(src, nParams, nParticles)
    v = (ub - lb)[:, None] * _rand(src, nParams, nParticles) * k
    bestX = X.copy()
    gX = bestX[:, 0].copy()
    bestF = np.full(nParticles, np.inf)
    Fv = np.full(nParticles, np.inf)
    globalbestF = np.inf
    startTime = time.process_time()
    nStalledIterations = 0
    if FitOpt.Verbosity:
        FitOpt.InfoPrintFunction('Starting iterations...')
    iIteration = 0
    stopCode = 0
    newbest = False
    elapsedTime = 0.0
    info = Struct()
    while stopCode == 0:
        iIteration += 1
        for p in range(nParticles):
            Fv[p], UserStop = _call(fcn, X[:, p].copy())
            if UserStop:
                stopCode = 2
                break
        idx = Fv < bestF
        bestF[idx] = Fv[idx]
        bestX[:, idx] = X[:, idx]
        idx = int(np.argmin(Fv))
        newbest = Fv[idx] < globalbestF
        if newbest:
            nStalledIterations = 0
            globalbestF = Fv[idx]
            gX = X[:, idx].copy()
        else:
            nStalledIterations += 1
        for p in range(nParticles):
            r1 = src.random(1)[0]
            r2 = src.random(1)[0]
            v[:, p] = w * v[:, p] + c1 * r1 * (bestX[:, p] - X[:, p]) + c2 * r2 * (gX - X[:, p])
            X[:, p] = X[:, p] + v[:, p]
            X[:, p] = np.minimum(np.maximum(X[:, p], lb), ub)
        if FitOpt.Verbosity:
            FitOpt.IterationPrintFunction('  Iteration %4d:   value %0.5e  best so far (%d)'
                                          % (iIteration, globalbestF, nStalledIterations))
        elapsedTime = (time.process_time() - startTime) / 60
        if elapsedTime > FitOpt.maxTime:
            stopCode = 1
        if globalbestF < FitOpt.TolFun:
            stopCode = 3
        if nStalledIterations == FitOpt.TolStallIter:
            stopCode = 4
        if not isempty(FitOpt.IterFcn):
            info.newbest = newbest
            if FitOpt.IterFcn(info):
                stopCode = 2
    if FitOpt.Verbosity > 0:
        msg = {1: 'Terminated: Time limit of %f minutes reached.' % FitOpt.maxTime,
               2: 'Terminated: Stopped by user.',
               3: 'Terminated: Found a parameter set with function value less than %g.' % FitOpt.TolFun,
               4: "Terminated: Function value didn't change for %d iterations." % FitOpt.TolStallIter}[stopCode]
        FitOpt.InfoPrintFunction(msg)
    info.nIterations = iIteration
    info.elapsedTime = elapsedTime
    info.newbest = newbest
    return gX, info
