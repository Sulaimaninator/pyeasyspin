"""MATLAB built-ins whose exact semantics the ports depend on.

One implementation shared by all functions. expm_eig_batch (end of the file) is EasySpin's
expm_ with algorithm 'eig', batched; it is not a MATLAB built-in.
"""
import math

import numpy as np

from ._mat import vec

EPS = np.finfo(float).eps


def round_half_away(x):
    """MATLAB round (half away from zero) for a scalar."""
    return math.copysign(math.floor(abs(x) + 0.5), x)


def colon(a, d, b):
    """MATLAB a:d:b as a 1-D array (MathWorks' colonop algorithm: symmetric about the midpoint,
    last element snapped to b within 2*eps*max(|a|,|b|))."""
    a, d, b = float(a), float(d), float(b)
    if not (np.isfinite(a) and np.isfinite(d) and np.isfinite(b)):
        return np.array([np.nan])
    if d == 0 or (a < b and d < 0) or (b < a and d > 0):
        return np.zeros(0)
    tol = 2.0 * EPS * max(abs(a), abs(b))
    sig = 1.0 if d > 0 else -1.0
    if a == math.floor(a) and d == 1:
        n = math.floor(b) - a
    elif a == math.floor(a) and d == math.floor(d):
        q = math.floor(a / d)
        r = a - q * d
        n = math.floor((b - r) / d) - math.floor((a - r) / d)
    else:
        n = round_half_away((b - a) / d)
        if sig * (a + n * d - b) > tol:
            n = n - 1
    n = int(n)
    last = a + n * d
    if sig * (last - b) > -tol:
        last = b
    out = np.zeros(n + 1)
    k = np.arange(0, n // 2 + 1)
    out[k] = a + k * d
    out[n - k] = last - k * d
    if n % 2 == 0:
        out[n // 2] = (a + last) / 2
    return out


def linspace(d1, d2, n):
    """MATLAB linspace(d1, d2, n) (its rounding: d1 + (k*(d2-d1))/(n-1), last = d2)."""
    n = int(math.floor(n))
    if n < 1:
        return np.zeros(0)
    if n == 1:
        return np.array([float(d2)])
    n1 = n - 1
    y = d1 + (np.arange(n1 + 1) * (d2 - d1)) / n1
    if d1 == d2:
        y[:] = d1
    else:
        y[-1] = d2
    return y


def nextpow2(n):
    n = abs(n)
    return 0 if n == 0 else int(math.ceil(math.log2(n)))


def cumtrapz(x, y=None):
    """MATLAB cumtrapz for vectors (1-D result)."""
    if y is None:
        y = vec(x)
        return np.concatenate(([0.0 * y[:1].sum()], np.cumsum(y[:-1] + y[1:]) / 2))
    x, y = vec(x), vec(y)
    dt = np.diff(x) / 2.0
    return np.concatenate(([0.0 * y[:1].sum()], np.cumsum(dt * (y[:-1] + y[1:]))))


def trapz(x, y):
    """MATLAB trapz(x, y) for vectors."""
    x, y = vec(x), vec(y)
    return np.dot(np.diff(x), (y[:-1] + y[1:])) / 2


def narrow(z):
    """MATLAB drops an all-zero imaginary part from arithmetic results."""
    z = np.asarray(z)
    if np.iscomplexobj(z) and not np.any(z.imag):
        return z.real.copy()
    return z


def cmax_index(z):
    """0-based index of MATLAB max() of a vector (complex: largest abs, ties -> largest angle)."""
    z = vec(z)
    if not np.iscomplexobj(z):
        return int(np.argmax(z))
    a = np.abs(z)
    cand = np.flatnonzero(a == a.max())
    return int(cand[np.argmax(np.angle(z[cand]))])


def interp1(x, v, xq, method='linear'):
    """MATLAB interp1(x, v, xq, method) for vector x, v; result has the shape of xq.
    'spline' = not-a-knot cubic spline, 'pchip' = Fritsch-Carlson (both extrapolate),
    'linear' gives NaN outside the data. Complex v: real and imaginary parts separately."""
    from scipy.interpolate import CubicSpline, PchipInterpolator
    x = vec(np.asarray(x, dtype=float))
    v = vec(np.asarray(v))
    xq_arr = np.asarray(xq, dtype=float)
    xqv = vec(xq_arr)
    if x.size > 1 and x[0] > x[-1]:
        x, v = x[::-1], v[::-1]
    if np.any(np.diff(x) < 0):
        o = np.argsort(x, kind='stable')
        x, v = x[o], v[o]

    def one(vr):
        if method == 'spline':
            return CubicSpline(x, vr, bc_type='not-a-knot')(xqv)
        if method == 'pchip':
            return PchipInterpolator(x, vr)(xqv)
        if method in ('linear', '*linear'):
            return np.interp(xqv, x, vr, left=np.nan, right=np.nan)
        raise ValueError(f'interp1 method {method} not supported')

    if np.iscomplexobj(v):
        if method == 'spline':
            out = one(v)
        else:
            out = one(v.real) + 1j * one(v.imag)
    else:
        out = one(v.astype(float))
    shape = xq_arr.shape if xq_arr.ndim >= 1 else (1,)
    return out.reshape(shape, order='F') if xq_arr.ndim >= 1 else out


def ppval(pp, x):
    """MATLAB ppval for a scalar-valued pp (breaks b, coefs c): the piece is located with
    [-inf, b(2:end-1), inf], so values beyond the ends use the end pieces; Horner in the local variable."""
    b, c = pp
    edges = b[1:-1]
    idx = np.searchsorted(edges, x, side='right')   # 0..pieces-1
    xs = x - b[idx]
    v = c[idx, 0].copy()
    for i in range(1, c.shape[1]):
        v = xs * v + c[idx, i]
    return v


def mkpp(breaks, coefs, d=1):
    """MATLAB mkpp: piecewise polynomial as a Struct with form, breaks (1 x l+1),
    coefs (d*l x k, column-major reshape of coefs), pieces, order and dim."""
    from ._struct import Struct
    breaks = np.asarray(breaks, dtype=float).ravel()
    coefs = np.asarray(coefs)
    l = breaks.size - 1
    dl = int(np.prod(np.atleast_1d(d))) * l
    k = int(np.fix(coefs.size / dl + 100 * EPS))
    if k <= 0 or k * dl != coefs.size:
        raise ValueError('mkpp: the number of coefficients does not match the breaks and the dimension.')
    return Struct(form='pp', breaks=breaks.reshape(1, -1), coefs=coefs.reshape((dl, k), order='F'),
                  pieces=float(l), order=float(k), dim=float(d) if np.ndim(d) == 0 else np.asarray(d, dtype=float))


def unmkpp(pp):
    """MATLAB unmkpp: (breaks, coefs, pieces, order, dim)."""
    return (np.asarray(pp.breaks), np.asarray(pp.coefs), int(pp.pieces), int(pp.order),
            int(np.asarray(pp.dim).ravel()[0]) if np.size(pp.dim) == 1 else np.asarray(pp.dim))


def fminsearch(fun, x0, TolX=1e-4, TolFun=1e-4, MaxFunEvals=None, MaxIter=None):
    """MATLAB fminsearch (Nelder-Mead, Lagarias et al.) with its default options and its
    initial simplex (5 % steps, 0.00025 for zero components)."""
    x0 = np.asarray(x0, dtype=float).ravel()
    n = x0.size
    maxfun = 200 * n if MaxFunEvals is None else MaxFunEvals
    maxiter = 200 * n if MaxIter is None else MaxIter
    rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5
    v = np.zeros((n, n + 1))
    fv = np.zeros(n + 1)
    v[:, 0] = x0
    fv[0] = fun(x0.copy())
    for j in range(n):
        y = x0.copy()
        y[j] = (1 + 0.05) * y[j] if y[j] != 0 else 0.00025
        v[:, j + 1] = y
        fv[j + 1] = fun(y.copy())
    o = np.argsort(fv, kind='stable')
    fv, v = fv[o], v[:, o]
    itercount = 1
    func_evals = n + 1
    while func_evals < maxfun and itercount < maxiter:
        if (np.max(np.abs(fv[0] - fv[1:])) <= max(TolFun, 10 * np.spacing(abs(fv[0])))
                and np.max(np.abs(v[:, 1:] - v[:, [0]])) <= max(TolX, 10 * np.spacing(abs(np.max(v[:, 0]))))):
            break
        xbar = np.sum(v[:, :n], axis=1) / n
        xr = (1 + rho) * xbar - rho * v[:, -1]
        fxr = fun(xr.copy())
        func_evals += 1
        if fxr < fv[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * v[:, -1]
            fxe = fun(xe.copy())
            func_evals += 1
            if fxe < fxr:
                v[:, -1], fv[-1] = xe, fxe
            else:
                v[:, -1], fv[-1] = xr, fxr
        else:
            if fxr < fv[n - 1]:
                v[:, -1], fv[-1] = xr, fxr
            else:
                shrink = False
                if fxr < fv[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * v[:, -1]
                    fxc = fun(xc.copy())
                    func_evals += 1
                    if fxc <= fxr:
                        v[:, -1], fv[-1] = xc, fxc
                    else:
                        shrink = True
                else:
                    xcc = (1 - psi) * xbar + psi * v[:, -1]
                    fxcc = fun(xcc.copy())
                    func_evals += 1
                    if fxcc < fv[-1]:
                        v[:, -1], fv[-1] = xcc, fxcc
                    else:
                        shrink = True
                if shrink:
                    for j in range(1, n + 1):
                        v[:, j] = v[:, 0] + sigma * (v[:, j] - v[:, 0])
                        fv[j] = fun(v[:, j].copy())
                    func_evals += n
        o = np.argsort(fv, kind='stable')
        fv, v = fv[o], v[:, o]
        itercount += 1
    return v[:, 0].copy()


# ---------------------------------------------------------------------------------------------
# Further MATLAB emulations, and EasySpin's expm_ 'eig' batched over a stack of matrices.
# The modules that use them import them under their old private names.


def bicgstab(A, b, tol, maxit):
    """MATLAB bicgstab(A, b, tol, maxit) without preconditioner and with x0 = 0 (R2024b
    algorithm: half-step convergence checks with the true residual, stagnation counter, extra
    steps, and the minimal-residual iterate returned on failure). Returns (x, flag)."""
    n = b.size
    eps = np.finfo(float).eps
    n2b = np.linalg.norm(b)
    if n2b == 0:
        return np.zeros(n, dtype=complex), 0
    x = np.zeros(n, dtype=complex)
    flag = 1
    xmin = x
    tolb = tol * n2b
    r = b - A @ x
    normr = np.linalg.norm(r)
    normr_act = normr
    if normr <= tolb:
        return x, 0
    rt = r
    normrmin = normr
    rho = 1
    omega = 1
    alpha = 0
    stag = 0
    moresteps = 0
    maxmsteps = min(n // 50, 10, n - maxit)
    maxstagsteps = 3
    p = v = None
    for ii in range(1, int(maxit) + 1):
        rho1 = rho
        rho = np.vdot(rt, r)
        if rho == 0 or np.isinf(rho):
            flag = 4
            break
        if ii == 1:
            p = r
        else:
            beta = (rho / rho1) * (alpha / omega)
            if beta == 0 or not np.isfinite(beta):
                flag = 4
                break
            p = r + beta * (p - omega * v)
        ph = p
        v = A @ ph
        rtv = np.vdot(rt, v)
        if rtv == 0 or np.isinf(rtv):
            flag = 4
            break
        alpha = rho / rtv
        if np.isinf(alpha):
            flag = 4
            break
        if abs(alpha) * np.linalg.norm(ph) < eps * np.linalg.norm(x):
            stag += 1
        else:
            stag = 0
        xhalf = x + alpha * ph
        s = r - alpha * v
        normr = np.linalg.norm(s)
        normr_act = normr
        if normr <= tolb or stag >= maxstagsteps or moresteps:
            s = b - A @ xhalf
            normr_act = np.linalg.norm(s)
            if normr_act <= tolb:
                x = xhalf
                flag = 0
                break
            if stag >= maxstagsteps and moresteps == 0:
                stag = 0
            moresteps += 1
            if moresteps >= maxmsteps:
                flag = 3
                x = xhalf
                break
        if stag >= maxstagsteps:
            flag = 3
            break
        if normr_act < normrmin:
            normrmin = normr_act
            xmin = xhalf
        sh = s
        t = A @ sh
        tt = np.vdot(t, t)
        if tt == 0 or np.isinf(tt):
            flag = 4
            break
        omega = np.vdot(t, s) / tt
        if np.isinf(omega):
            flag = 4
            break
        if abs(omega) * np.linalg.norm(sh) < eps * np.linalg.norm(xhalf):
            stag += 1
        else:
            stag = 0
        x = xhalf + omega * sh
        r = s - omega * t
        normr = np.linalg.norm(r)
        normr_act = normr
        if normr <= tolb or stag >= maxstagsteps or moresteps:
            r = b - A @ x
            normr_act = np.linalg.norm(r)
            if normr_act <= tolb:
                flag = 0
                break
            if stag >= maxstagsteps and moresteps == 0:
                stag = 0
            moresteps += 1
            if moresteps >= maxmsteps:
                flag = 3
                break
        if normr_act < normrmin:
            normrmin = normr_act
            xmin = x
        if stag >= maxstagsteps:
            flag = 3
            break
    if flag != 0:
        r_comp = b - A @ xmin
        if np.linalg.norm(r_comp) <= normr_act:
            x = xmin
    return x, flag


class Cell(list):
    """An N-D MATLAB cell array: a list of its elements in column-major order (so it compares
    like a recorded cell) plus its MATLAB shape `mshape`. Assignment with 1-based subscripts grows
    it as MATLAB does (new elements are [])."""

    def __init__(self, items=(), mshape=None):
        super().__init__(items)
        self.mshape = tuple(mshape) if mshape is not None else ((1, len(self)) if len(self) else (0, 0))

    @staticmethod
    def _trim(shape):
        shape = list(shape)
        while len(shape) > 2 and shape[-1] == 1:
            shape.pop()
        return tuple(shape)

    def _lin(self, idx, shape):
        lin, stride = 0, 1
        for k, i in enumerate(idx):
            lin += (i - 1) * stride
            stride *= shape[k] if k < len(shape) else 1
        return lin

    def setitem(self, idx, value):
        idx = [int(i) for i in idx]
        if len(idx) == 1:
            k = idx[0]
            if self.mshape == (0, 0) or self.mshape[0] == 1:
                idx = [1, k]
            elif self.mshape[1] == 1 and len(self.mshape) == 2:
                idx = [k, 1]
            else:
                if k > len(self):
                    raise IndexError('A(I) = X: X must have the same size as I.')
                list.__setitem__(self, k - 1, value)
                return
        old = list(self.mshape) + [1] * (len(idx) - len(self.mshape))
        new = [max(o, i) for o, i in zip(old, idx + [1] * (len(old) - len(idx)))]
        if new != old:
            items = [np.zeros((0, 0)) for _ in range(int(np.prod(new)))]
            for lin in range(len(self)):
                sub, r = [], lin
                for o in old:
                    sub.append(r % o + 1)
                    r //= o
                items[self._lin(sub, new)] = list.__getitem__(self, lin)
            self[:] = items
            self.mshape = self._trim(new)
        list.__setitem__(self, self._lin(idx, list(self.mshape) + [1] * len(idx)), value)

    def getitem(self, idx):
        idx = [int(i) for i in idx]
        if len(idx) == 1:
            return list.__getitem__(self, idx[0] - 1)
        return list.__getitem__(self, self._lin(idx, list(self.mshape) + [1] * len(idx)))


def round_sig(x, n):
    """MATLAB round(x, n, 'significant')."""
    if x == 0 or not np.isfinite(x):
        return x
    digits = n - int(math.floor(math.log10(abs(x)))) - 1
    if digits >= 0:
        p = 10.0 ** digits
        return round_half_away(x * p) / p
    p = 10.0 ** (-digits)
    return round_half_away(x / p) * p


def smooth3_gaussian(data, sz=(3, 3, 3), sd=0.65):
    """MATLAB smooth3(data, 'gaussian', sz) (default standard deviation 0.65): normalised Gaussian
    kernel on the grid -(sz-1)/2:(sz-1)/2, replicate padding, convn 'valid'."""
    data = np.asarray(data, dtype=float)
    ax = [np.arange(-(s - 1) / 2, (s - 1) / 2 + 0.5) for s in sz]
    x, y, z = np.meshgrid(*ax, indexing='ij')
    h = np.exp(-(x * x / 2 / sd ** 2 + y * y / 2 / sd ** 2 + z * z / 2 / sd ** 2))
    h = h / np.sum(h)
    pad = [((s - 1) // 2, (s - 1) // 2) for s in sz]
    P = np.pad(data, pad, mode='edge')
    out = np.zeros(data.shape)
    n1, n2, n3 = data.shape
    for i in range(sz[0]):          # convn with a symmetric kernel = correlation
        for j in range(sz[1]):
            for k in range(sz[2]):
                out += h[sz[0] - 1 - i, sz[1] - 1 - j, sz[2] - 1 - k] * P[i:i + n1, j:j + n2, k:k + n3]
    return out


def discretize(x, edges):
    """MATLAB discretize(x, edges): 1-based bin index (bins [e_k, e_k+1), the last one closed), NaN outside."""
    e = np.asarray(edges, dtype=float).ravel(order='F')
    k = np.searchsorted(e, x, side='right').astype(float)    # number of edges <= x
    k[x == e[-1]] = e.size - 1
    k[(x < e[0]) | (x > e[-1]) | np.isnan(x)] = np.nan
    return k


def griddedinterpolant_linear(grids, values):
    """MATLAB griddedInterpolant(grids, values, 'linear') (linear extrapolation) as a callable on N x 3 points."""
    from scipy.interpolate import RegularGridInterpolator
    return RegularGridInterpolator(tuple(grids), values, method='linear', bounds_error=False, fill_value=None)


def cumtrapz_rows(t, y):
    """MATLAB cumtrapz(t, y, 2) for a matrix y."""
    dt = np.diff(t) / 2.0
    return np.concatenate([np.zeros((y.shape[0], 1)), np.cumsum(dt * (y[:, :-1] + y[:, 1:]), axis=1)], axis=1)


def expm_eig_batch(A):
    """expm_(A, 'eig') for a stack of matrices (..., n, n): Hermitian -> eigh, else eig, then
    V*diag(exp(d))*V' (as EasySpin's expm_ 'eig', one LAPACK call per matrix)."""
    n = A.shape[-1]
    flat = A.reshape(-1, n, n)
    out = np.empty(flat.shape, dtype=complex)
    herm = np.array([np.array_equal(M, M.conj().T) for M in flat])
    if np.any(herm):
        import scipy.linalg
        for k in np.flatnonzero(herm):
            d, V = scipy.linalg.eigh(flat[k])
            out[k] = V @ np.diag(np.exp(d)) @ V.conj().T
    nh = np.flatnonzero(~herm)
    if nh.size:
        d, V = np.linalg.eig(flat[nh])
        out[nh] = (V * np.exp(d)[:, None, :]) @ np.conj(np.swapaxes(V, -1, -2))
    return out.reshape(A.shape)
