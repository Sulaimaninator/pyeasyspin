"""Given an interaction tensor and one or more trajectories of rotation matrices, calculate the
tensor trajectories"""
import numpy as np
from .._registry import F


def _mshape(shape):
    """MATLAB drops trailing singleton dimensions beyond the second."""
    shape = list(shape)
    while len(shape) > 2 and shape[-1] == 1:
        shape.pop()
    return tuple(shape)


def cardamom_tensortraj(T=None, RTraj=None, RTrajInv=None):
    """cardamom_tensortraj(T, RTraj, RTrajInv) -> TTraj = RTraj * T * RTrajInv page-wise; T is a 3-vector
    (principal values) or a 3x3 matrix, RTraj is 3x3xnSteps[xnTraj]."""
    T = np.asarray(T)
    if T.ndim <= 2 and min(T.shape + (1,) * (2 - T.ndim)) == 1 and T.size == 3:
        T = np.diag(T.ravel(order='F'))
    elif T.ndim == 2 and T.shape == (3, 3):
        pass
    else:
        raise ValueError('Interaction tensor T needs to be given as a 3-vector or 3x3 matrix.')
    RTraj = np.asarray(RTraj)
    if RTraj.ndim < 2 or RTraj.shape[0] != 3 or RTraj.shape[1] != 3:
        raise ValueError('Rotation matrix trajectory needs to have size 3x3xnTrajxnSteps.')
    nTraj = RTraj.shape[3] if RTraj.ndim > 3 else 1
    nSteps = RTraj.shape[2] if RTraj.ndim > 2 else 1
    Trep = np.broadcast_to(T.reshape(3, 3, 1, 1), (3, 3, nSteps, nTraj)).reshape(RTraj.shape)   # repmat
    return F.multimatmult(RTraj, F.multimatmult(np.ascontiguousarray(Trep), RTrajInv))
