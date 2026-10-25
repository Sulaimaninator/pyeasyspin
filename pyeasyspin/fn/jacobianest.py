"""Jacobian estimation by Romberg-extrapolated finite differences (private; after D'Errico's
DERIVESTsuite, as in EasySpin)."""
import numpy as np
from scipy.linalg import solve_triangular

from .._registry import F, nout
from .._mat import isempty


def _call(fun, x):
    """MATLAB [f,~,UserStop] = fun(x) for functions returning a value or a tuple."""
    r = fun(x)
    if isinstance(r, tuple):
        return np.asarray(r[0]), (bool(r[2]) if len(r) > 2 else False)
    return np.asarray(r), False


def _swapelement(vec, ind, val):
    v = np.array(vec, dtype=float, copy=True)
    v[np.unravel_index(ind, v.shape, order='F')] = val
    return v


def _romberg_extrapolation(StepRatio, der_init, rombexpon):
    """Vectorized over rows of der_init (n x nsteps): returns der_romb, errest (n x nest)."""
    srinv = 1 / StepRatio
    rombexpon = np.asarray(rombexpon, dtype=float)
    nexpon = rombexpon.size
    rmat = np.ones((nexpon + 2, nexpon + 1))
    rmat[1, 1:3] = srinv**rombexpon
    rmat[2, 1:3] = srinv**(2 * rombexpon)
    rmat[3, 1:3] = srinv**(3 * rombexpon)
    qromb, rromb = np.linalg.qr(rmat)
    ne = der_init.shape[1]
    m = ne - (nexpon + 2)
    ind = np.arange(nexpon + 2)[:, None] + np.arange(m)[None, :]      # vec2mat
    rhs = der_init[:, ind]                                             # n x (nexpon+2) x m
    rombcoefs = solve_triangular(rromb, np.einsum('ki,nkm->nim', qromb, rhs).transpose(1, 0, 2).reshape(nexpon + 1, -1))
    rombcoefs = rombcoefs.reshape(nexpon + 1, rhs.shape[0], m).transpose(1, 0, 2)   # n x (nexpon+1) x m
    der_romb = rombcoefs[:, 0, :]
    s = np.sqrt(np.sum((rhs - np.einsum('ki,nim->nkm', rmat, rombcoefs))**2, axis=1))
    rinv = solve_triangular(rromb, np.eye(nexpon + 1))
    cov1 = np.sum(rinv**2, axis=1)
    errest = s * 12.7062047361747 * np.sqrt(cov1[0])
    return der_romb, errest


@nout(2)
def jacobianest(fun=None, x0=None, MaxRelStep=None, lb=None, ub=None):
    """jacobianest(fun, x0, MaxRelStep=0.01, lb=None, ub=None) -> (jac, err).

    fun returns the residual vector (or a tuple whose first element is it and whose third is a
    user-stop flag). lb/ub (Python extension, DEVIATIONS.md): when the central stencil
    x0(i) +- delta would leave [lb, ub], a one-sided stencil towards the side with room is used
    (Romberg exponents [1 2] instead of [2 4]), so the model is never evaluated outside the
    bounds. Without lb/ub, or when the stencil fits, the result is EasySpin's."""
    if MaxRelStep is None:
        MaxRelStep = 0.01
    StepRatio = 2 + 1e-7
    nSteps = 10
    if isinstance(fun, str):
        fun = F.get(fun)
    x0 = np.asarray(x0, dtype=float)
    nx = x0.size
    MaxRelStep = np.asarray(MaxRelStep, dtype=float).reshape(-1)
    if MaxRelStep.size == 1:
        MaxRelStep = MaxRelStep[0] * np.ones(nx)
    lb = None if isempty(lb) else np.asarray(lb, dtype=float).reshape(-1, order='F')
    ub = None if isempty(ub) else np.asarray(ub, dtype=float).reshape(-1, order='F')
    f0, _ = _call(fun, x0)
    f0 = f0.reshape(-1, order='F')
    n = f0.size
    if n == 0:
        jac = np.zeros((0, nx))
        return jac, jac.copy()
    steps = StepRatio**np.arange(0, -nSteps - 1, -1, dtype=float)
    nsteps = steps.size
    jac = np.zeros((n, nx), dtype=f0.dtype if np.iscomplexobj(f0) else float)
    err = np.zeros((n, nx))
    x0flat = x0.reshape(-1, order='F')
    for i in range(nx):
        relativedelta = MaxRelStep[i] * steps
        x0_i = x0flat[i]
        delta = x0_i * relativedelta if x0_i != 0 else relativedelta
        side = 0
        if lb is not None or ub is not None:
            dmax = abs(delta[0])
            room_lo = x0_i - lb[i] if lb is not None else np.inf
            room_hi = ub[i] - x0_i if ub is not None else np.inf
            if room_lo < dmax or room_hi < dmax:
                side, room = (1, room_hi) if room_hi >= room_lo else (-1, room_lo)
                delta = np.abs(delta) * (min(room, dmax) / dmax)
        fdel = np.zeros((n, nsteps), dtype=jac.dtype)
        for j in range(nsteps):
            if side == 0:
                fp, _ = _call(fun, _swapelement(x0, i, x0_i + delta[j]))
                fm, UserStop = _call(fun, _swapelement(x0, i, x0_i - delta[j]))
                if UserStop:
                    return np.zeros((0, 0)), np.zeros((0, 0))
                fdel[:, j] = fp.reshape(-1, order='F') - fm.reshape(-1, order='F')
            else:
                fs, UserStop = _call(fun, _swapelement(x0, i, x0_i + side * delta[j]))
                if UserStop:
                    return np.zeros((0, 0)), np.zeros((0, 0))
                fdel[:, j] = fs.reshape(-1, order='F') - f0
        if side == 0:
            derest = fdel * (0.5 / delta)[None, :]
            rombexpon = [2, 4]
        else:
            derest = fdel / (side * delta)[None, :]
            rombexpon = [1, 2]
        der_romb, errest = _romberg_extrapolation(StepRatio, derest, rombexpon)
        nest = der_romb.shape[1]
        tags = np.argsort(der_romb, axis=1, kind='stable')
        keep = tags[:, 3:nest - 3]                       # trim = [1:3, nest+(-2:0)]
        rows = np.arange(n)[:, None]
        ek = errest[rows, keep]
        k = np.argmin(ek, axis=1)
        sel = keep[np.arange(n), k]
        err[:, i] = errest[np.arange(n), sel]
        jac[:, i] = der_romb[np.arange(n), sel]
    return jac, err
