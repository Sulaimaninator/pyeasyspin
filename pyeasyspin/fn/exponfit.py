"""Fits exponential(s) to 1D and 2D data"""
import numpy as np
from .._registry import nout
from .._mat import vec

_EPS = np.finfo(float).eps


def _lsq(A, b):
    return np.linalg.lstsq(A, b, rcond=None)[0]


@nout(3)
def exponfit(x=None, yData=None, nExponents=None, noconst=None):
    """exponfit(x, y, nExponents=1, noconst='-') -> (k, c, yFit): y = c0 + c1*exp(-k1*x) [+ c2*exp(-k2*x)]
    per column (or the row of a row vector); 'noconst' drops c0. Variable projection +
    Levenberg-Marquardt (EasySpin's mexpfit2)."""
    if x is None or yData is None:
        raise ValueError('Wrong number of input arguments!')
    if nExponents is None:
        nExponents = 1
    if noconst is None:
        noconst = '-'
    nExponents = int(nExponents)
    if nExponents not in (1, 2):
        raise ValueError('Number of exponents must be 1 or 2.')
    includeOffset = noconst != 'noconst'
    Options = [includeOffset, 1e-11, 1e-11, 1e4, 1e-3]
    yData = np.asarray(yData, dtype=float)
    if yData.ndim < 2:
        yData = yData.reshape(1, -1)
    flipped = yData.shape[0] == 1
    if flipped:
        yData = yData.T
    nRows, nCols = yData.shape
    x = vec(np.asarray(x, dtype=float))
    scale = np.max(np.abs(x))
    x = x / scale
    k = np.zeros((nExponents, nCols))
    c = np.zeros((nExponents + 1, nCols))
    yFitted = np.zeros((nRows, nCols))
    for iCol in range(nCols):
        ySlice = yData[:, iCol].copy()
        isRecovery = ySlice[0] < ySlice[-1]
        if isRecovery:
            ySlice = -ySlice
        kStart = np.array([-_guessSingleDecay(x, ySlice)])
        if nExponents == 2:
            kStart = np.array([0.8, 1.3]) * kStart
        kk, c_ = _mexpfit2(x, ySlice, kStart, Options)
        k[:, iCol] = kk
        if not includeOffset:
            c[0, iCol] = 0
            c[1:, iCol] = c_
        else:
            c[:, iCol] = c_
        if isRecovery:
            c[:, iCol] = -c[:, iCol]
        yFitted[:, iCol] = c[0, iCol] + c[1, iCol] * np.exp(k[0, iCol] * x)
        if nExponents == 2:
            yFitted[:, iCol] += c[2, iCol] * np.exp(k[1, iCol] * x)
    k = k / scale
    if flipped:
        yFitted = yFitted.T
    else:
        k = k.T
        c = c.T
    return -k, c, yFitted


def _guessSingleDecay(t, y):
    C = y[-1]
    A = y[0] - C
    dt = t[1] - t[0]
    integral_ = np.sum(y - C) * dt
    kGuessed = A / integral_
    if kGuessed < 0:
        kGuessed = 1 / (t.size / 2 * dt)
    return kGuessed


def _mexpfit2(tData, yData, xGuess, Options):
    x = -np.sort(-vec(xGuess))
    n = x.size
    if np.any(x >= 0):
        raise ValueError('k must be strictly negative')
    IncludeOffset = Options[0]
    MinGradNorm = Options[1]
    MaxFunEvals = Options[3]
    Differences, Jacobian, _ = _func(x, tData, yData, IncludeOffset)
    A = Jacobian.T @ Jacobian
    Gradient = Jacobian.T @ Differences
    F = (Differences @ Differences) / 2
    GradientNorm = np.linalg.norm(Gradient, np.inf)
    mu = Options[4] * np.max(np.diag(A))
    nFunEvals = 1
    nu = 2
    Stop = 0
    mok = 0
    while not Stop:
        if GradientNorm <= MinGradNorm:
            Stop = 1
        elif nFunEvals == MaxFunEvals:
            Stop = 3
        else:
            Step = np.linalg.solve(A + mu * np.eye(n), -Gradient)
            NormStep = np.linalg.norm(Step)
            nx = Options[2] + np.linalg.norm(x)
            if NormStep <= Options[2] * nx:
                Stop = 2
            elif NormStep >= nx / _EPS:
                Stop = 4
        if not Stop:
            i = np.nonzero(x + Step >= 0)[0]
            if i.size:
                Step = np.min(-0.99 * x[i] / Step[i]) * Step
                mu = 10 * mu
            xnew = x + Step
            Step = xnew - x
            dL = (Step @ (mu * Step - Gradient)) / 2
            NewDifferences, NewJacobian, _ = _func(xnew, tData, yData, IncludeOffset)
            nFunEvals += 1
            newF = (NewDifferences @ NewDifferences) / 2
            dF = F - newF
            if dL > 0 and dF > 0:
                mok += 1
                mu = mu * max(1 / 3, 1 - (2 * dF / dL - 1) ** 3)
                nu = 2
                x = xnew
                F = newF
                Jacobian = NewJacobian
                Differences = NewDifferences
                A = Jacobian.T @ Jacobian
                Gradient = Jacobian.T @ Differences
                GradientNorm = np.linalg.norm(Gradient, np.inf)
            else:
                mu = mu * nu
                nu = 2 * nu
                if mok > n and nu > 8:
                    Stop = 5
    x = -np.sort(-x)
    _, _, c = _func(x, tData, yData, IncludeOffset)
    if IncludeOffset:
        c = c[np.r_[n, 0:n]]
    return x, c


def _func(x, tData, yData, IncludeOffset):
    m = tData.size
    nExp = x.size
    E = np.exp(np.outer(tData, x))
    A = np.hstack([E, np.ones((m, 1))]) if IncludeOffset else E
    c = _lsq(A, yData)
    Residuals = yData - A @ c
    Jacobian = np.zeros((m, nExp))
    AA = A.T @ A
    for j in range(nExp):
        jj = -tData * A[:, j]
        b = c[j] * (A.T @ jj)
        b[j] = b[j] - jj @ Residuals
        Jacobian[:, j] = c[j] * jj - A @ np.linalg.solve(AA, b)
    return Residuals, Jacobian, c
