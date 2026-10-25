"""Bound-constrained minimisation with L-BFGS-B, [x, f, info] = mdhmm_lbfgsb(fcn, l, u, opts)"""
import math
import numpy as np
from .._registry import F, nout
from .._struct import Struct

_TASKSTR = {209: 'ERROR: N .LE. 0', 210: 'ERROR: M .LE. 0', 211: 'ERROR: FACTR .LT. 0',
            3: 'ABNORMAL_TERMINATION_IN_LNSRCH.', 4: 'RESTART_FROM_LNSRCH.',
            21: 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL.',
            22: 'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH.', 31: 'STOP: CPU EXCEEDING THE TIME LIMIT.',
            32: 'STOP: TOTAL NO. of f AND g EVALUATIONS EXCEEDS LIM.', 33: 'STOP: THE PROJECTED GRADIENT IS SUFFICIENTLY SMALL.',
            101: 'WARNING: ROUNDING ERRORS PREVENT PROGRESS', 102: 'WARNING: XTOL TEST SATISIED',
            103: 'WARNING: STP = STPMAX', 104: 'WARNING: STP = STPMIN', 201: 'ERROR: STP .LT. STPMIN',
            202: 'ERROR: STP .GT. STPMAX', 203: 'ERROR: INITIAL G .GE. ZERO ', 204: 'ERROR: FTOL .LT. ZERO',
            205: 'ERROR: GTOL .LT. ZERO', 206: 'ERROR: XTOL .LT. ZERO', 207: 'ERROR: STPMIN .LT. ZERO',
            208: 'ERROR: STPMAX .LT. STPMIN', 212: 'ERROR: INVALID NBD', 213: 'ERROR: NO FEASIBLE SOLUTION'}


@nout(3)
def mdhmm_lbfgsb(fcn=None, l=None, u=None, opts=None):
    """mdhmm_lbfgsb(fcn, l, u, opts=None) -> (x, f, info). fcn(x) returns (f, g) (or a pair [F, G] of
    callables); l, u are column vectors of bounds (+-Inf = none). opts fields as in EasySpin: x0, m (5),
    factr (1e7), pgtol (1e-5), maxIts (100), maxTotalIts (5000), printEvery, errFcn. The optimiser is
    SciPy's L-BFGS-B (mdhmm_lbfgsb_wrapper); info has iterations, totalIterations, lbfgs_message1
    and err (per-iteration f and ||g||_inf)."""
    if fcn is None or l is None or u is None:
        raise ValueError('Not enough input arguments.')
    opts = Struct() if opts is None or (isinstance(opts, (list, np.ndarray)) and len(opts) == 0) else Struct(opts)
    if len(opts) == 0:
        opts = Struct(a=1)

    def setOpts(field, default, mn=None, mx=None):
        if field not in opts:
            opts[field] = default
        out = opts[field]
        if mn is not None and np.any(np.asarray(out) < mn):
            raise ValueError('Value is too small')
        if mx is not None and np.any(np.asarray(out) > mx):
            raise ValueError('Value is too large')
        del opts[field]
        return out

    if isinstance(fcn, (list, tuple)):
        Ff, Gf = fcn[0], fcn[1]
        callF = lambda x: (Ff(x), Gf(x))
    else:
        callF = fcn
    l = np.asarray(l, dtype=float)
    u = np.asarray(u, dtype=float)
    l = l.reshape(-1, 1) if l.ndim < 2 else l
    u = u.reshape(-1, 1) if u.ndim < 2 else u
    n = max(l.shape)
    if max(u.shape) != max(l.shape):
        raise ValueError('l and u must be same length')
    x0 = np.asarray(setOpts('x0', np.zeros((n, 1))), dtype=float)
    x0 = x0.reshape(-1, 1) if x0.ndim < 2 else x0
    x = x0 + 0
    if x0.shape[1] != 1:
        raise ValueError('x0 must be a column vector')
    if l.shape[1] != 1:
        raise ValueError('l must be a column vector')
    if u.shape[1] != 1:
        raise ValueError('u must be a column vector')
    if x.shape[0] != n:
        raise ValueError('x0 and l have mismatchig sizes')
    if u.shape[0] != n:
        raise ValueError('u and l have mismatchig sizes')
    m = setOpts('m', 5, 0)
    nbd = np.isfinite(l).astype(int) + np.isfinite(u).astype(int) + 2 * (np.isinf(l) & np.isfinite(u)).astype(int)
    factr = setOpts('factr', 1e7, 0)
    pgtol = setOpts('pgtol', 1e-5, 0)
    maxIts = setOpts('maxIts', 100, 1)
    maxTotalIts = setOpts('maxTotalIts', 5e3)
    printEvery = setOpts('printEvery', math.inf)
    errFcn = setOpts('errFcn', [])
    iprint = setOpts('verbose', -1)
    history = {}

    def callF_wrapped(xx, outerIter0):
        f, g = callF(xx)
        outerIter = int(outerIter0) + 1
        history[outerIter] = (float(f), float(np.max(np.abs(np.asarray(g, dtype=float)))) if np.size(g) else 0.0)
        if callable(errFcn):
            history[outerIter] = history[outerIter] + (float(errFcn(xx)),)
        if not math.isinf(printEvery) and outerIter > 1 and (outerIter - 1) % int(printEvery) == 0:
            h = history[outerIter - 1] if (outerIter - 1) in history else history[outerIter]
            print('Iter %5d, f(x) = %2e, ||grad||_infty = %.2e' % (outerIter - 1, h[0], h[1]))
        return f, g

    f, x, taskInteger, outer_count, k = F.mdhmm_lbfgsb_wrapper(m, x, l, u, nbd, callF_wrapped, factr, pgtol,
                                                               iprint, maxIts, maxTotalIts)
    info = Struct()
    info.iterations = outer_count
    info.totalIterations = k
    info.lbfgs_message1 = _TASKSTR.get(int(taskInteger), 'UNRECOGNIZED EXIT FLAG')
    kmax = max(history) if history else 0
    width = len(next(iter(history.values()))) if history else 2
    err = np.zeros((kmax, width))
    for i, h in history.items():
        err[i - 1, :len(h)] = h
    info.err = err
    return np.asarray(x, dtype=float).reshape(-1, 1), f, info
