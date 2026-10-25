"""Compute rotation matrix from rotation axis and rotation angle"""
import numpy as np
from .._registry import F


def rotaxi2mat(n=None, rho=None):
    """rotaxi2mat(n (3-vector or 'x','y',...), rho) -> R (3x3)."""
    if rho is None:
        raise ValueError('Provide a second input argument (rotation angle rho).')
    if np.iscomplexobj(rho) or np.size(rho) != 1:
        raise ValueError('Second input (angle rho) must be a real number.')
    rho = float(np.asarray(rho).ravel()[0])
    if isinstance(n, str):
        n = F.letter2vec(n)
    elif np.size(n) != 3 or np.iscomplexobj(n):
        raise ValueError("Rotation axis must be a 3-element vector or a shortcut such as 'x', 'y', etc.")
    n = np.asarray(n, dtype=float).ravel(order='F')
    n = n / np.linalg.norm(n)
    N = np.zeros((3, 3))
    N[1, 2], N[2, 0], N[0, 1] = n[0], n[1], n[2]
    N = N - N.T
    R = np.eye(3) + N * np.sin(rho) + (N @ N) * (1 - np.cos(rho))
    thresh = 1e-10
    R[np.abs(R) < thresh] = 0
    R[np.abs(R - 1) < thresh] = 1
    R[np.abs(R + 1) < thresh] = -1
    return R
