"""Interpolation of data given on an EasySpin orientation grid (sphgrid) to new angles."""
import numpy as np

from .._registry import F
from .._mat import vec
from .._matlab import colon, linspace, interp1


def _fastlinearinterp1d(y, xx):
    y = vec(y)
    k = np.minimum(np.maximum(1 + np.floor(xx - 1), 1), y.size - 1).astype(int)
    return y[k - 1] + (xx - k) * (y[k] - y[k - 1])


def _rectify(y, GridSize, nOctants, periodic, cubicInterp):
    y = vec(y)
    fullSphere = nOctants == 8
    if fullSphere:
        nr, nc, dlen = 2 * GridSize - 1, 4 * (GridSize - 1) + 1, 4
    else:
        nr, nc, dlen = GridSize, nOctants * (GridSize - 1) + 1, nOctants
    z = np.zeros((nr, nc), dtype=y.dtype)
    z[0, :] = y[0]
    idx = 2
    length = dlen + 1 - int(periodic)

    def row(idx, length):
        sel = list(range(idx, idx + length)) + ([idx] if periodic else [])
        seg = y[np.array(sel) - 1]
        if cubicInterp:
            return F.esspline1d(seg.reshape(1, -1), 2 if periodic else 1, float(nc)).ravel()
        return _fastlinearinterp1d(seg, linspace(1, length, nc))

    for ir in range(2, GridSize):
        z[ir - 1, :] = row(idx, length)
        idx += length
        length += dlen
    sel = list(range(idx, idx + length)) + ([idx] if periodic else [])
    z[GridSize - 1, :] = y[np.array(sel) - 1]
    if fullSphere:
        idx += length
        length -= dlen
        for ir in range(GridSize + 1, 2 * GridSize - 1):
            z[ir - 1, :] = row(idx, length)
            idx += length
            length -= dlen
        z[2 * GridSize - 2, :] = y[idx - 1]
    return z


def _interp2_linear(z, xi, yi):
    """MATLAB interp2(Z, XI, YI, 'linear') on the index grid; NaN outside."""
    nr, nc = z.shape
    xi, yi = np.asarray(xi, dtype=float), np.asarray(yi, dtype=float)
    out = np.full(np.broadcast(xi, yi).shape, np.nan, dtype=np.result_type(z.dtype, float))
    X, Y = np.broadcast_arrays(xi, yi)
    ok = (X >= 1) & (X <= nc) & (Y >= 1) & (Y <= nr)
    x, y_ = X[ok], Y[ok]
    c = np.minimum(np.floor(x).astype(int), nc - 1) if nc > 1 else np.ones(x.size, dtype=int)
    r = np.minimum(np.floor(y_).astype(int), nr - 1) if nr > 1 else np.ones(y_.size, dtype=int)
    tx, ty = x - c, y_ - r
    c1 = np.minimum(c + 1, nc)
    r1 = np.minimum(r + 1, nr)
    z00, z01 = z[r - 1, c - 1], z[r - 1, c1 - 1]
    z10, z11 = z[r1 - 1, c - 1], z[r1 - 1, c1 - 1]
    # zero-weight corners are left out (not 0*value): on a grid line MATLAB returns the grid value
    # even when the neighbouring cell holds NaN
    def term(w, v):
        return np.where(w == 0, 0.0, w * v)

    out[ok] = (term((1 - ty) * (1 - tx), z00) + term((1 - ty) * tx, z01)
               + term(ty * (1 - tx), z10) + term(ty * tx, z11))
    return out


def gridinterp(y=None, gridParams=None, phi=None, theta=None, InterpType=None):
    """Interpolated values at (phi, theta). InterpType 'G3' (global cubic, default), 'L3' (local
    cubic, 1-D grids only) or 'G1'/'L1' (linear)."""
    if theta is None and phi is None:
        raise ValueError('4 or 5 input arguments ar required.')
    if InterpType is None or len(InterpType) == 0:
        InterpType = 'G3'
    globalInterpolation = InterpType[0].upper() == 'G'
    cubicInterpolation = InterpType[1] == '3'
    gp = vec(np.asarray(gridParams, dtype=float))
    nOrientations, GridSize = int(gp[0]), int(gp[1])
    periodic = not bool(gp[2])
    nOctants, maxPhi = int(gp[3]), gp[4]
    y = np.atleast_2d(np.asarray(y))
    if max(y.shape) != nOrientations:
        raise ValueError('Wrong number of points. Not a valid data set!')
    if nOctants == 0:
        m, n = y.shape
        phi_a = np.asarray(phi)
        Factor = (phi_a.size - 1) / (n - 1)
        iphi = colon(1, 1 / Factor, n)
        if not cubicInterpolation:
            return interp1(np.arange(1.0, n + 1), y.T, iphi) if m == 1 else \
                np.column_stack([interp1(np.arange(1.0, n + 1), y[r, :], iphi) for r in range(m)])
        if globalInterpolation:
            return F.esspline1d(y, 1, iphi)
        Fac = int(round(Factor))
        x = np.arange(0, Fac) / Factor
        X = np.column_stack([x ** 3, x ** 2, x, np.ones(Fac)])
        H = np.array([[2.0, -2.0, 1.0, 1.0], [-3.0, 3.0, -2.0, -1.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        yi = []
        for r in range(m):
            dl = np.diff(y[r, :])
            k = np.flatnonzero(np.sign(dl[0:n - 2]) * np.sign(dl[1:n - 1]) > 0)
            dmax = np.maximum(np.abs(dl[k]), np.abs(dl[k + 1]))
            dmin = np.minimum(np.abs(dl[k]), np.abs(dl[k + 1]))
            T = np.zeros(n)
            T[k + 1] = 2 * dmin * dmax / (dl[k] + dl[k + 1])
            C = np.vstack([y[r, 0:-1], y[r, 1:], T[0:-1], T[1:]])
            yii = X @ H @ C
            yi.append(np.concatenate([yii.ravel(order='F'), [y[r, n - 1]]]))
        return np.array(yi)
    z = _rectify(y, GridSize, nOctants, periodic, cubicInterpolation)
    phi_a, theta_a = np.asarray(phi, dtype=float), np.asarray(theta, dtype=float)
    if nOctants < 8:
        iphi = 1 + nOctants * (GridSize - 1) * (phi_a / maxPhi)
        ithe = 1 + (GridSize - 1) * (theta_a / (np.pi / 2))
    else:
        iphi = 1 + 4 * (GridSize - 1) * (phi_a / maxPhi)
        ithe = 1 + 2 * (GridSize - 1) * (theta_a / np.pi)
    if not cubicInterpolation:
        return _interp2_linear(z, iphi, ithe)
    if globalInterpolation:
        return F.esspline2d(z, ithe, iphi)
    raise ValueError(f'Local cubic interpolator for {nOctants} (oct) not available!')
