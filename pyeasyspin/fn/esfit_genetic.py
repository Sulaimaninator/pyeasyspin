"""Genetic algorithm for least-squares fitting"""
import math
import time

import numpy as np

from .._registry import F, nout
from .._struct import Struct
from .._mat import isempty


def _call(fcn, x):
    r = fcn(x)
    return float(np.real(r[0] if isinstance(r, tuple) else r))


def _rand(src, m, n):
    return src.random(m * n).reshape((m, n), order='F')


def _randi(src, lo, hi, n):
    """MATLAB randi([lo hi],1,n) (integers lo..hi inclusive); the draws differ from MATLAB's."""
    f = getattr(src, 'integers', None) or src.randint
    return np.asarray(f(lo, hi + 1, n))


@nout(2)
def esfit_genetic(fcn=None, lb=None, ub=None, FitOpt=None, rng=None):
    """esfit_genetic(fcn, lb, ub, FitOpt) -> (bestx, info). Random draws come from rng (numpy
    Generator/RandomState; default the global numpy stream). bestx is 1-D.
    """
    if fcn is None:
        return None
    if lb is None or ub is None:
        raise ValueError('At least 3 inputs expected (function, lb, ub).')
    if FitOpt is None:
        FitOpt = Struct()
    DefOpt, _ = F.esfit_algdefaults('genetic algorithm')
    FitOpt = F.adddefaults(FitOpt, DefOpt)
    if isempty(FitOpt.get('EliteCount')):
        FitOpt.EliteCount = max(2, math.ceil(0.1 * FitOpt.PopulationSize))
    src = np.random if rng is None else rng
    stopCode = 0
    lb = np.asarray(lb, dtype=float).reshape(-1, order='F')
    ub = np.asarray(ub, dtype=float).reshape(-1, order='F')
    if lb.size != ub.size:
        raise ValueError('Arrays for lower and upper bound must have the same number of elements.')
    if np.any(lb > ub):
        raise ValueError('Lower bounds must not be greater than upper bounds.')
    nParams = lb.size
    P = int(FitOpt.PopulationSize)
    Elite = int(FitOpt.EliteCount)
    if FitOpt.Verbosity:
        FitOpt.InfoPrintFunction(['  %d parameters' % nParams,
                                  '  population %d, elite %d' % (P, Elite),
                                  '  %d generations' % FitOpt.maxGenerations])
    nEvals = 0
    startTime = time.process_time()
    Population = lb + (ub - lb) * _rand(src, P, nParams)
    bestScore = np.inf
    bestx = np.zeros(nParams)
    if FitOpt.Verbosity:
        FitOpt.IterationPrintFunction('initial population')
    info = Struct()
    Scores = np.full(P, np.inf)
    for k in range(P):
        Scores[k] = _call(fcn, Population[k, :].copy())
        nEvals += 1
        newbest = Scores[k] < bestScore
        if newbest:
            bestx = Population[k, :].copy()
            bestScore = Scores[k]
        info.bestx = bestx
        info.minF = bestScore
        info.nEvals = nEvals
        info.iter = 0
        info.newbest = newbest
        UserStop = bool(FitOpt.IterFcn(info)) if not isempty(FitOpt.IterFcn) else False
        if UserStop:
            stopCode = 3
            break
    idx = np.argsort(Scores, kind='stable')
    Scores = Scores[idx]
    Population = Population[idx, :]
    Fitness = Scores[-1] - Scores
    gen = 1
    offScores = np.zeros(P)
    while True:
        if stopCode:
            break
        if np.min(Scores) < bestScore:
            bestScore = np.min(Scores)
        if FitOpt.Verbosity:
            FitOpt.IterationPrintFunction('gen %5d:  min %g   mean %g' % (gen, np.min(Scores), np.mean(Scores)))
        if gen >= FitOpt.maxGenerations:
            stopCode = 0
            break
        elapsedTime = (time.process_time() - startTime) / 60
        if elapsedTime > FitOpt.maxTime:
            stopCode = 1
            break
        if bestScore < FitOpt.TolFun:
            stopCode = 2
            break
        RouletteWheel = np.concatenate(([0.0], np.cumsum(Fitness) / np.sum(Fitness)))
        Balls = (0.5 + np.arange(P)) / P
        ParentsIdx = np.array([np.sum(Balls[k] > RouletteWheel) for k in range(P)])   # 1-based
        r = _rand(src, 2, P)
        r = ParentsIdx[np.fix(P * r + 1).astype(int) - 1] - 1                         # 0-based rows
        d = 0.1
        a = _rand(src, 1, P)[0] * (1 + 2 * d) - d
        Offspring = np.empty((P, nParams))
        for k in range(P):
            Offspring[k, :] = a[k] * Population[r[0, k], :] + (1 - a[k]) * Population[r[1, k], :]
        nk = 10
        rr = 2 * _randi(src, 0, 1, P) - 1
        jj = _randi(src, 1, nk - 1, P)
        for k in range(P):
            rangelim = np.vstack([lb, Offspring[k, :]]) if rr[k] == -1 else np.vstack([Offspring[k, :], ub])
            delta = (rangelim[1, :] - rangelim[0, :]) / nk
            Offspring[k, :] = Offspring[k, :] + rr[k] * _rand(src, 1, nParams)[0] * delta * jj[k]
        for p in range(nParams):
            Offspring[Offspring[:, p] < lb[p], p] = lb[p]
            Offspring[Offspring[:, p] > ub[p], p] = ub[p]
        for k in range(P):
            offScores[k] = _call(fcn, Offspring[k, :].copy())
            nEvals += 1
            info.nEvals = nEvals
            info.iter = gen
            UserStop = bool(FitOpt.IterFcn(info)) if not isempty(FitOpt.IterFcn) else False
            if UserStop:
                stopCode = 3
                break
        if stopCode == 3:
            break
        idx = np.argsort(offScores, kind='stable')
        offScores = offScores[idx]
        Offspring = Offspring[idx, :]
        idx = np.arange(P - Elite)
        Population[idx + Elite, :] = Offspring[idx, :]
        Scores[idx + Elite] = offScores[idx]
        idx = np.argsort(Scores, kind='stable')
        Scores = Scores[idx]
        Population = Population[idx, :]
        newbest = Scores[0] < bestScore
        if newbest:
            bestScore = Scores[0]
            bestx = Population[0, :].copy()
        Fitness = Scores[-1] - Scores
        info.bestx = bestx
        info.minF = bestScore
        info.nEvals = nEvals
        info.iter = gen
        info.newbest = newbest
        UserStop = bool(FitOpt.IterFcn(info)) if not isempty(FitOpt.IterFcn) else False
        if UserStop:
            stopCode = 3
            break
        gen += 1
    if FitOpt.Verbosity > 0:
        msg = {0: 'Maximum number of generations (%d) reached.' % FitOpt.maxGenerations,
               1: 'Time limit of %f minutes reached.' % FitOpt.maxTime,
               2: 'Error below threshold %g.' % FitOpt.TolFun,
               3: 'Stopped by user.'}[stopCode]
        FitOpt.InfoPrintFunction('Terminated: %s\n' % msg)
    return bestx, info
