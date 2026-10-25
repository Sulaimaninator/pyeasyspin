"""Convert cartesian vectors to polar angles"""
import numpy as np
from .._registry import F, nout


def _sc(a):
    return float(a.ravel()[0]) if a.size == 1 else a


@nout(2)
def vec2ang(*varargin, nargout=None):
    """vec2ang(v (3xN, Nx3 or 'x'...)) or vec2ang(x, y, z) -> (phi, theta);
    nargout=1 -> [phi; theta] stacked."""
    nargin = len(varargin)
    if nargin == 1:
        v = varargin[0]
        if isinstance(v, str):
            v = F.letter2vec(v)
        v = np.asarray(v)
        if v.ndim == 1:
            v = v.reshape(-1, 1)
        if v.shape[0] != 3 and v.shape[1] == 3:
            v = v.T
        if v.shape[0] != 3:
            raise ValueError('At least one dimension of the input array must be 3.')
        if not np.issubdtype(v.dtype, np.number) or np.iscomplexobj(v):
            raise ValueError('Argument must contain real numbers!')
        v = v.astype(float).reshape(3, -1, order='F')
        x, y, z = v[0:1], v[1:2], v[2:3]
    elif nargin == 3:
        x, y, z = (np.atleast_2d(np.asarray(a, dtype=float)) for a in varargin)
        if x.size != y.size or x.size != z.size:
            raise ValueError('The three inputs (x,y,z) must be real-valued arrays of the same size.')
    else:
        raise ValueError('Either 1 or 3 inputs expected.')
    phi = np.arctan2(y, x)
    theta = np.pi / 2 - np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    if nargout in (0, 1):
        return np.concatenate([phi, theta], axis=0)
    return _sc(phi), _sc(theta)
