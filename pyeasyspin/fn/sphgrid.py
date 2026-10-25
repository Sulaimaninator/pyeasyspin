"""Spherical grid"""
import numpy as np
from scipy.spatial import Delaunay
from .._registry import F, nout
from .._struct import Struct
from .._matlab import linspace as _ml_linspace   # MATLAB's rounding: the azimuth sign at phi = pi depends on it


def _sphgrid_(nOctants, maxPhi, GridSize, closedPhi):
    dtheta = (np.pi / 2) / (GridSize - 1)
    if nOctants > 0:
        nOct = 4 if nOctants == 8 else nOctants
        nOri = int(GridSize + nOct * GridSize * (GridSize - 1) // 2)
        phi = np.zeros(nOri)
        theta = np.zeros(nOri)
        W = np.zeros(nOri)
        sindth2 = np.sin(dtheta / 2)
        w0 = 0.5 if closedPhi else 1.0
        W[0] = maxPhi * (1 - np.cos(dtheta / 2))
        Start = 1
        for iSlice in range(2, GridSize):
            nPhi = nOct * (iSlice - 1) + 1
            dPhi = maxPhi / (nPhi - 1)
            idx = Start + np.arange(nPhi)
            W[idx] = 2 * np.sin((iSlice - 1) * dtheta) * sindth2 * dPhi * np.r_[w0, np.ones(nPhi - 2), 0.5]
            phi[idx] = _ml_linspace(0, maxPhi, nPhi)
            theta[idx] = (iSlice - 1) * dtheta
            Start = Start + nPhi
        nPhi = nOct * (GridSize - 1) + 1
        dPhi = maxPhi / (nPhi - 1)
        idx = Start + np.arange(nPhi)
        phi[idx] = _ml_linspace(0, maxPhi, nPhi)
        theta[idx] = np.pi / 2
        W[idx] = sindth2 * dPhi * np.r_[w0, np.ones(nPhi - 2), 0.5]
        if not closedPhi:
            rmv = np.cumsum(nOct * np.arange(1, GridSize) + 1)   # 1-based cumsum(...)+1 -> 0-based
            keep = np.ones(nOri, bool)
            keep[rmv] = False
            phi, theta, W = phi[keep], theta[keep], W[keep]
        if nOctants == 8:
            idx = np.arange(len(theta) - nPhi, -1, -1)           # MATLAB length-nPhi+1:-1:1
            phi = np.r_[phi, phi[idx]]
            theta = np.r_[theta, np.pi - theta[idx]]
            W[idx] = W[idx] / 2
            W = np.r_[W, W[idx]]
        W = 2 * (2 * np.pi / maxPhi) * W
    elif nOctants == 0:
        phi = np.zeros(GridSize)
        theta = _ml_linspace(0, np.pi / 2, GridSize)
        c = np.r_[0, dtheta / 2 + dtheta * np.arange(GridSize - 1), np.pi / 2]
        W = -2 * (2 * np.pi) * np.diff(np.cos(c))
    elif nOctants == -1:
        phi = np.zeros(1)
        theta = np.zeros(1)
        W = np.array([4 * np.pi])
    else:
        raise ValueError(f'Unsupported value {nOctants} for nOctants.')
    return phi, theta, W


def _triangleareas(Tri, vecs):
    x1 = vecs[:, Tri[:, 0] - 1]
    x2 = vecs[:, Tri[:, 1] - 1]
    x3 = vecs[:, Tri[:, 2] - 1]
    a1 = np.arccos(np.sum(x2 * x3, axis=0) + 0j)
    a2 = np.arccos(np.sum(x3 * x1, axis=0) + 0j)
    a3 = np.arccos(np.sum(x1 * x2, axis=0) + 0j)
    s = (a1 + a2 + a3) / 2
    x = np.sqrt(np.tan(s / 2) * np.tan((s - a1) / 2) * np.tan((s - a2) / 2) * np.tan((s - a3) / 2) + 0j)
    x = np.real(x)
    return 4 * np.arctan(x)


def _delaunay(x, y):
    return Delaunay(np.column_stack([x, y])).simplices.astype(np.int64) + 1


@nout(2)
def sphgrid(Symmetry=None, GridSize=None, Options=None, nargout=None):
    """sphgrid(Symmetry or nOctants, GridSize[, 'c']) -> grid Struct; nargout=2 -> (grid, tri).
    The triangulation uses scipy's Delaunay (Qhull) in place of MATLAB's delaunay."""
    if GridSize is None:
        raise ValueError('Wrong number of input arguments!')
    if Options is None:
        Options = ''
    if nargout is None:
        nargout = 1
    explicitClosedPhi = 'c' in Options
    calcTri = nargout > 1
    if isinstance(Symmetry, str):
        maxPhi, closedPhi, nOctants = F.gridparam(Symmetry)
    else:
        nOctants = int(Symmetry)
        maxPhi = {-1: 0.0, 0: 0.0, 1: np.pi / 2, 2: np.pi, 4: 2 * np.pi, 8: 2 * np.pi}.get(nOctants)
        if maxPhi is None:
            raise ValueError('Wrong number of octants, must be -1, 0, 1, 2, 4, or 8.')
        closedPhi = False
    if explicitClosedPhi:
        closedPhi = True
    if nOctants > 1 and closedPhi:
        raise ValueError(f'Cannot provide closed-phi grid for this symmetry ({nOctants} octants).')
    if np.size(GridSize) != 1 or float(GridSize) < 2 or float(GridSize) % 1:
        raise ValueError('GridSize (2nd input argument) must be an integer >= 2.')
    GridSize = int(GridSize)
    phi, theta, Weights = _sphgrid_(nOctants, maxPhi, GridSize, closedPhi)
    phi, theta, Weights = phi[None, :], theta[None, :], Weights[None, :]
    vecs = np.asarray(F.ang2vec(phi, theta))
    grid = Struct()
    grid.Symmetry = Symmetry
    grid.GridSize = GridSize
    grid.closedPhi = bool(closedPhi)
    grid.nOctants = nOctants
    grid.maxPhi = maxPhi
    grid.phi = phi
    grid.theta = theta
    grid.vecs = vecs
    grid.weights = Weights
    grid.n = Weights.size
    if not calcTri:
        return grid
    G = GridSize
    p, t = phi.ravel(), theta.ravel()
    if nOctants in (0, -1):
        Tri = np.zeros((0, 3), np.int64)
        Areas = np.zeros(0)
    elif nOctants == 1 and closedPhi:
        a = np.arange(1, G * (G + 1) // 2 + 1)
        a = np.delete(a, (np.arange(2, G + 2) * np.arange(1, G + 1)) // 2 - 1)
        k = np.arange(1, (G - 1) * (G - 2) // 2 + 1)
        b = a[k - 1]
        Tri = np.hstack([np.vstack([np.arange(1, G * (G - 1) // 2 + 1), a, a + 1]),
                         np.vstack([b, 2 * (b + 1) - k, b + 1])]).T
        Areas = _triangleareas(Tri, vecs)
    elif nOctants in (1, 2):
        phx = np.r_[p, maxPhi * np.ones(G - 1)]
        thx = np.r_[t, np.pi / 2 * np.arange(1, G) / (G - 1)]
        Tri = _delaunay(thx * np.cos(phx), thx * np.sin(phx))
        vecsx = np.asarray(F.ang2vec(phx, thx))
        Areas = _triangleareas(Tri, vecsx)
        rmv = np.arange(p.size + 1, phx.size + 1)
        if nOctants == 1:
            rpl = (np.arange(0, G - 1) * np.arange(1, G)) // 2 + 2
        else:
            rpl = np.arange(0, G - 1) ** 2 + np.arange(0, G - 1) + 2
        for kk in range(rmv.size):
            Tri[Tri == rmv[kk]] = rpl[kk]
    elif nOctants == 4:
        Tri = _delaunay(t * np.cos(p), t * np.sin(p))
        Areas = _triangleareas(Tri, vecs)
    elif nOctants == 8:
        phx, thx, _ = _sphgrid_(4, 2 * np.pi, G, False)
        triUpper = _delaunay(thx * np.cos(phx), thx * np.sin(phx))
        nTotal = 4 * G ** 2 - 8 * G + 6
        triLower = nTotal + 1 - triUpper
        nEquator = 4 * (G - 1)
        idxEq = triUpper > thx.size - nEquator
        triLower[idxEq] = triUpper[idxEq]
        Tri = np.vstack([triUpper, triLower])
        Areas = _triangleareas(Tri, vecs)
    else:
        raise ValueError('Triangulation for this symmetry not supported!')
    if nOctants >= 1 and not closedPhi:
        rmv = Areas < np.mean(Areas) * 1e-3
        Tri = Tri[~rmv]
        Areas = Areas[~rmv]
    Tri = np.sort(Tri, axis=1).astype(np.uint32)
    if Tri.shape[0]:
        idx = np.lexsort(Tri.T[::-1])
        Tri = Tri[idx]
        Areas = Areas[idx]
    Areas = Areas / np.sum(Areas) * 4 * np.pi if Areas.size else Areas
    tri = Struct()
    tri.idx = Tri
    tri.areas = Areas.reshape(-1, 1)
    return grid, tri
