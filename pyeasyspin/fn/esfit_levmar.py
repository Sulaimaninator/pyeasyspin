"""Levenberg-Marquardt nonlinear least squares fitting"""

import numpy as np
from scipy.linalg import solve_triangular

from .._registry import F, nout
from .._struct import Struct
from .._mat import isempty
from .._parallel import prefetch, cpu_time


def _call(fcn, x):
    """MATLAB [f,~,userstop] = fcn(x) for functions returning a vector or a tuple."""
    r = fcn(x)
    if isinstance(r, tuple):
        return np.asarray(r[0]), (bool(r[2]) if len(r) > 2 else False)
    return np.asarray(r), False


def _rand(rng, n):
    return (np.random if rng is None else rng).random(n)


# Keep finite-difference steps inside [lb, ub] (DEVIATIONS.md). False reproduces EasySpin's
# always-forward step exactly (used by the parity tests).
BOUNDED_FD = True


def _fdstep(x, ix, delta, lb, ub):
    """Forward step x(ix)+delta; backward if that leaves the upper bound and the backward step
    stays inside (bounded finite differences, DEVIATIONS.md)."""
    if BOUNDED_FD and x[ix] + delta > ub[ix] and x[ix] - delta >= lb[ix]:
        return -delta
    return delta


@nout(2)
def esfit_levmar(fcn=None, x0=None, lb=None, ub=None, FitOpt=None, rng=None):
    """esfit_levmar(fcn, x0, lb, ub, FitOpt) -> (x, info). fcn(x) returns the residual vector
    (or a tuple (residuals, rmsd, userstop)). Finite-difference Jacobian steps are kept inside
    [lb, ub] (DEVIATIONS.md); otherwise as EasySpin."""
    if fcn is None:
        return None
    if x0 is None or lb is None or ub is None:
        raise ValueError('At least 4 inputs expected (function, x0, lb, ub).')
    if FitOpt is None:
        FitOpt = Struct()
    DefOpt, _ = F.esfit_algdefaults('Levenberg-Marquardt')
    FitOpt = F.adddefaults(FitOpt, DefOpt)
    delta = FitOpt.delta
    startTime = cpu_time()
    Fval = np.nan
    norm_g = np.nan
    nEvals = 0
    lb = np.asarray(lb, dtype=float).reshape(-1, order='F')
    ub = np.asarray(ub, dtype=float).reshape(-1, order='F')
    if lb.size != ub.size:
        raise ValueError('Arrays for lower and upper bound must have the same number of elements.')
    if np.any(lb > ub):
        raise ValueError('Lower bounds must not be greater than upper bounds.')
    x0 = np.asarray(x0, dtype=float).reshape(-1, order='F')
    n = x0.size
    if np.any(np.isnan(x0)) or np.any(np.isinf(x0)):
        raise ValueError('x0 must be real and finite.')
    if np.any(x0 < lb) or np.any(x0 > ub):
        raise ValueError('Some elements in x0 are out of bounds.')
    nParams = x0.size
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
        fcn.prefetch = lambda X: prefetch(fcn0, [untransform(x) for x in X])
    x = x0.copy()
    stopCode = 0
    Je = None
    info = Struct()
    stopCode, Fval, f = _funeval(fcn, x)
    nEvals += 1
    if not stopCode:
        Je, stopCode = _JacobianEstimate(fcn, x, f, delta, lb, ub)
        nEvals += n
        if not stopCode:
            g = Je.T @ f
            norm_g = np.linalg.norm(g, np.inf)
            A = Je.T @ Je
            if np.isinf(norm_g) or np.isinf(np.max(np.abs(A))):
                stopCode = -5
    if stopCode:
        info.F = Fval
        info.norm_g = norm_g
        info.Je = Je
        info.stop = stopCode
        info.nEvals = nEvals
        return x, info

    mu = FitOpt['lambda'] * np.max(np.diag(A))
    nu = 2
    norm_h = 0
    j = 0
    nEvals = 0
    iIteration = 0
    while not stopCode:
        iIteration += 1
        if norm_g <= FitOpt.Gradient:
            stopCode = 1
            break
        h, mu = _computeLMStep(A, g, mu)
        norm_h = np.linalg.norm(h)
        if FitOpt.Verbosity:
            FitOpt.IterationPrintFunction(' iteration %4d:  value %0.5e    gradient %0.5e    step %0.5e'
                                          % (iIteration, np.sqrt(Fval * 2), norm_g, norm_h))
        if norm_h <= FitOpt.TolStep * (FitOpt.TolStep + np.linalg.norm(x)):
            stopCode = 2
            break
        xnew = x + h
        xnew = np.minimum(np.maximum(xnew, lb), ub)
        # The trial point and, when this iteration will need it, the Jacobian-update point below
        # are independent: evaluate them together.
        jn = j % n + 1
        if abs(h[jn - 1]) < 0.8 * norm_h:
            xu = x.copy()
            xu[jn - 1] = x[jn - 1] + _fdstep(x, jn - 1, delta, lb, ub)
            prefetch(fcn, [xnew, xu])
        stopCode, Fnew, fnew = _funeval(fcn, xnew)
        nEvals += 1
        if stopCode:
            break
        info.bestx = xnew
        info.minF = Fnew
        info.nEvals = nEvals
        info.iter = iIteration
        info.newbest = True
        UserStop = bool(FitOpt.IterFcn(info)) if not isempty(FitOpt.IterFcn) else False
        if UserStop:
            stopCode = 4
            break
        j = j % n + 1
        gamma = 0.8
        if abs(h[j - 1]) < gamma * norm_h:
            xu = x.copy()
            xu[j - 1] = x[j - 1] + _fdstep(x, j - 1, delta, lb, ub)
            stopCode, _, fu = _funeval(fcn, xu)
            nEvals += 1
            if not stopCode:
                hu = xu - x
                Je = Je + np.outer((fu - f - Je @ hu) / (hu @ hu), hu)
        Je = Je + np.outer((fnew - f - Je @ h) / (h @ h), h)
        rho = (Fval - Fnew) / (0.5 * (h @ (mu * h - g)))
        if rho > 0:
            x = xnew
            Fval = Fnew
            f = fnew
        if rho > 0:
            mu = mu * max(1 / 3, 1 - (2 * rho - 1)**3)
            nu = 2
        else:
            mu = mu * nu
            nu = 2 * nu
        g = Je.T @ f
        norm_g = np.linalg.norm(g, np.inf)
        A = Je.T @ Je
        if np.isinf(norm_g) or np.isinf(np.max(np.abs(A))):
            stopCode = -5
            break
        elapsedTime = (cpu_time() - startTime) / 60
        if elapsedTime > FitOpt.maxTime:
            stopCode = 3
            break
    if stopCode and iIteration == 1:
        info.bestx = x
        info.minF = Fval
        info.nEvals = nEvals
        info.iter = iIteration
        info.newbest = True
        if not isempty(FitOpt.IterFcn):
            FitOpt.IterFcn(info)
    if stopCode < 0:
        FitOpt['lambda'] = np.nan
    else:
        FitOpt['lambda'] = mu / np.max(np.diag(A))
    msg = {1: 'Gradient below threshold of %g' % FitOpt.Gradient,
           2: 'Parameter step below threshold of %g' % FitOpt.TolStep,
           3: 'Time limit of %f minutes reached' % FitOpt.maxTime,
           4: 'Stopped by user'}.get(stopCode, f'stop code {stopCode}')
    if FitOpt.Verbosity > 0:
        FitOpt.InfoPrintFunction('Terminated: %s\n' % msg)
    if transformParams:
        x = untransform(x)
    info.F = Fval
    info.norm_g = norm_g
    info.norm_h = norm_h
    info.Je = Je
    info['lambda'] = FitOpt['lambda']
    info.nIterations = iIteration - 1
    info.stop = stopCode
    info.nEvals = nEvals
    return x, info


def _JacobianEstimate(funfcn, x0, f0, delta, lb, ub):
    nVariables = x0.size
    J = np.zeros((f0.size, nVariables), dtype=f0.dtype if np.iscomplexobj(f0) else float)
    pts = []
    for ix in range(nVariables):
        x1 = x0.copy()
        x1[ix] = x0[ix] + _fdstep(x0, ix, delta, lb, ub)
        pts.append(x1)
    prefetch(funfcn, pts)                            # all columns at once
    for ix in range(nVariables):
        x1 = x0.copy()
        d = _fdstep(x0, ix, delta, lb, ub)
        x1[ix] = x0[ix] + d
        f1, userstop = _call(funfcn, x1)
        if userstop:
            return J, 5
        f1 = f1.reshape(-1, order='F')
        J[:, ix] = (f1 - f0) / d
    if np.iscomplexobj(J) or np.any(np.isnan(J)) or np.any(np.isinf(J)):
        err = -6
    else:
        err = 0
    return J, err


def _computeLMStep(A, g, mu):
    notPosDef = True
    while notPosDef:
        try:
            R = np.linalg.cholesky(A + mu * np.eye(A.shape[0])).T
            notPosDef = False
        except np.linalg.LinAlgError:
            notPosDef = True
        if not notPosDef:
            notPosDef = 1 / np.linalg.cond(R, 1) < 1e-15
        if notPosDef:
            mu = 10 * mu
    h = solve_triangular(R, solve_triangular(R, -g, trans='T'))
    return h, mu


def _funeval(funfcn, x):
    errCode = 0
    f, stopCode = _call(funfcn, x)
    if stopCode:
        errCode = 4
    f = f.reshape(-1, order='F')
    if np.any(np.isnan(f)):
        raise ValueError('f contains at least one NaN value.')
    if np.any(np.isinf(f)):
        raise ValueError('f contains at least one Inf value.')
    if np.iscomplexobj(f) and np.any(np.imag(f) != 0):
        raise ValueError('f is not real-valued.')
    f = np.real(f).astype(float)
    Fv = f @ f
    if np.isinf(Fv):
        errCode = -5
    return errCode, Fv, f
