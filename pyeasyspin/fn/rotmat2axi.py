"""Rotation axis and angle form rotation matrix"""
import numpy as np
from .._registry import nout


@nout(2)
def rotmat2axi(R=None):
    """rotmat2axi(R) -> (n (3x1), rho)."""
    R = np.asarray(R)
    if not np.issubdtype(R.dtype, np.number) or np.iscomplexobj(R) or R.shape != (3, 3):
        raise ValueError('R must be a 3x3 real matrix')
    R = R.astype(float)
    if np.trace(R) == 3:
        return np.array([[0.], [0.], [1.]]), 0.0
    rho = float(np.real(np.arccos(complex((np.trace(R) - 1) / 2))))
    n = np.array([[R[1, 2] - R[2, 1]], [R[2, 0] - R[0, 2]], [R[0, 1] - R[1, 0]]])
    return n / np.linalg.norm(n), rho
