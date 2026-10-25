"""Draw random samples from a 3-variate probability distribution using rejection sampling"""
import numpy as np
from .._registry import nout


def _randi_range(rng, lohi, size):
    """MATLAB randi([imin imax], ...) when called with a size vector [imin imax] (here size(grid))."""
    lo, hi = int(lohi[0]), int(lohi[1])
    if lo > hi:
        raise ValueError('First input must be a positive scalar integer value IMAX, or two integer values [IMIN IMAX] with IMIN less than or equal to IMAX.')
    if hasattr(rng, 'integers'):
        return rng.integers(lo, hi + 1, size)
    return rng.randint(lo, hi + 1, size)


@nout(3)
def cardamom_rejectionsample3d(pdf=None, xGrid=None, yGrid=None, zGrid=None, nSamples=None, rng=None):
    """cardamom_rejectionsample3d(pdf, xGrid, yGrid, zGrid, nSamples) -> (xSamples, ySamples, zSamples),
    each 1 x nSamples. Random: proposals and acceptance draws come from rng (numpy Generator or
    RandomState; default the global numpy stream), drawn in batches, so the samples follow the same
    distribution as MATLAB's but not its sequence. As in EasySpin, the grid index proposal uses
    randi(size(grid)), i.e. randi([1 n]) for a row vector grid."""
    rng = np.random if rng is None else rng
    pdf = np.asarray(pdf, dtype=float)
    xGrid, yGrid, zGrid = (np.asarray(g, dtype=float) for g in (xGrid, yGrid, zGrid))
    sz = [g.shape if g.ndim >= 2 else (1, g.size) for g in (xGrid, yGrid, zGrid)]
    nSamples = int(nSamples)
    c = np.max(pdf)
    xs, ys, zs = (np.zeros((1, nSamples)) for _ in range(3))
    xv, yv, zv = (g.ravel(order='F') for g in (xGrid, yGrid, zGrid))
    have = 0
    while have < nSamples:
        nb = max(1000, 2 * (nSamples - have))
        ix = _randi_range(rng, sz[0], nb)
        iy = _randi_range(rng, sz[1], nb)
        iz = _randi_range(rng, sz[2], nb)
        u = rng.random(nb)
        acc = u < pdf[ix - 1, iy - 1, iz - 1] / c
        k = np.flatnonzero(acc)[:nSamples - have]
        xs[0, have:have + k.size] = xv[ix[k] - 1]
        ys[0, have:have + k.size] = yv[iy[k] - 1]
        zs[0, have:have + k.size] = zv[iz[k] - 1]
        have += k.size
    return xs, ys, zs
