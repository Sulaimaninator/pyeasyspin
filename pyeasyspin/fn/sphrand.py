"""Generate random points over unit sphere"""
import numpy as np
from .._registry import nout


@nout(-1)
def sphrand(N=None, nOctants=None, nargout=None, rng=None):
    """sphrand(N[, nOctants=4]) -> 3xN unit vectors; nargout=2 -> (phi, theta); nargout=3 -> (x, y, z).
    Uses numpy's random generator (rng keyword), so values differ from MATLAB's rand stream."""
    if nOctants is None:
        nOctants = 4
    rng = np.random.default_rng() if rng is None else rng
    N = int(N)
    nOctants = int(nOctants)
    if nOctants in (1, 2, 4):
        z = rng.random((1, N))
        phi = nOctants * np.pi / 2 * rng.random((1, N))
    elif nOctants == 8:
        z = 2 * rng.random((1, N)) - 1
        phi = 2 * np.pi * rng.random((1, N))
    else:
        raise ValueError(f'Number of octants is {nOctants}, but must be 1, 2, 4 or 8!')
    if nargout is None:
        nargout = 1
    if nargout == 2:
        return phi, np.arccos(z)
    r = np.sqrt(1 - z ** 2)
    if nargout in (0, 1):
        return np.vstack([r * np.cos(phi), r * np.sin(phi), z])
    if nargout == 3:
        return r * np.cos(phi), r * np.sin(phi), z
    raise ValueError('Wrong number of output arguments!')
