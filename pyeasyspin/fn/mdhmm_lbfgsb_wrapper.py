"""L-BFGS-B driver (replaces EasySpin's mex gateway to the L-BFGS-B 3.0 Fortran/C code by
Stephen Becker with SciPy's L-BFGS-B, which implements the same algorithm, version 3.0)."""
import numpy as np
import scipy.optimize
from .._registry import nout

# taskInteger codes of the mex (mdhmm_lbfgsb.inc / findTaskString)
_TASK = (('NORM OF PROJECTED GRADIENT', 21), ('NORM_OF_PROJECTED_GRADIENT', 21),
         ('PROJECTED GRADIENT IS SUFFICIENTLY SMALL', 33), ('REL REDUCTION OF F', 22), ('REL_REDUCTION_OF_F', 22),
         ('ABNORMAL', 3), ('EVALUATIONS EXCEEDS', 32), ('ITERATIONS REACHED LIMIT', 2), ('NO FEASIBLE', 213),
         ('ROUNDING ERRORS', 101))


@nout(5)
def mdhmm_lbfgsb_wrapper(m=None, x=None, l=None, u=None, nbd=None, fcn=None, factr=1e7, pgtol=1e-5,
                         iprint=-1, maxIts=100, maxTotalIts=5000):
    """mdhmm_lbfgsb_wrapper(m, x, l, u, nbd, fcn, factr, pgtol, iprint, maxIts, maxTotalIts)
    -> (f, x, taskInteger, outer_count, k). fcn(x, iteration) returns (f, g). Minimises with
    scipy.optimize.minimize(method='L-BFGS-B') using maxcor=m, ftol=factr*eps, gtol=pgtol,
    maxiter=maxIts, maxfun=maxTotalIts; nbd 0/1/2/3 = unbounded/lower/both/upper as in L-BFGS-B.
    k counts setulb calls as the mex does (function evaluations + iterations + 1). taskInteger is
    the mex's code: 21/22 convergence, 2 iteration limit (NEW_X), 3 abnormal line search, 32
    evaluation limit."""
    x0 = np.asarray(x, dtype=float).ravel(order='F')
    n = x0.size
    l = np.asarray(l, dtype=float).ravel(order='F')
    u = np.asarray(u, dtype=float).ravel(order='F')
    nbd = np.asarray(nbd).ravel(order='F').astype(int)
    bounds = []
    for i in range(n):
        lo = l[i] if nbd[i] in (1, 2) else None
        hi = u[i] if nbd[i] in (2, 3) else None
        bounds.append((lo, hi))
    state = {'iter': 0}

    def fun(xx):
        f, g = fcn(xx.reshape(-1, 1), state['iter'])
        return float(f), np.asarray(g, dtype=float).ravel(order='F')

    def cb(*a):
        state['iter'] += 1

    res = scipy.optimize.minimize(fun, x0, jac=True, method='L-BFGS-B', bounds=bounds, callback=cb,
                                  options=dict(maxcor=int(m), ftol=float(factr) * np.finfo(float).eps,
                                               gtol=float(pgtol), maxiter=int(maxIts), maxfun=int(maxTotalIts),
                                               maxls=20))
    msg = res.message if isinstance(res.message, str) else res.message.decode()
    task = next((code for key, code in _TASK if key.lower() in msg.lower()), 0)
    outer = int(res.nit)
    k = int(res.nfev) + outer + 1
    return float(res.fun), res.x.reshape(-1, 1), float(task), float(outer), float(k)
