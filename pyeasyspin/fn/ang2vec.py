"""Polar angles to cartesian coordinates"""
import numpy as np
from .._registry import nout


@nout(-1)
def ang2vec(phi=None, theta=None, nargout=None):
    """ang2vec(phi, theta) -> 3xN array; nargout=3 -> (x, y, z) as 1xN rows."""
    if nargout is None:
        nargout = 1
    if nargout not in (0, 1, 3):
        raise ValueError('One or three output arguments are required.')
    phi = np.asarray(phi, dtype=float).ravel(order='F')
    theta = np.asarray(theta, dtype=float).ravel(order='F')
    nphi, ntheta = phi.size, theta.size
    if nphi != 1 and ntheta != 1 and nphi != ntheta:
        raise ValueError('phi and theta must have the same number of elements!')
    if nphi == 1:
        phi = np.repeat(phi, ntheta)
    elif ntheta == 1:
        theta = np.repeat(theta, nphi)
    sintheta = np.sin(theta)
    x = sintheta * np.cos(phi)
    y = sintheta * np.sin(phi)
    z = np.cos(theta)
    if nargout == 3:
        return x[None, :], y[None, :], z[None, :]
    return np.vstack([x, y, z])
