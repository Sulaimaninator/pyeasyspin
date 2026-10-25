"""Rotate a frame around a given axis"""
import numpy as np
from .._registry import F, nout


@nout(2)
def rotateframe(ang0=None, nRot=None, rho=None):
    """rotateframe(ang0, nRot, rho) -> (ang (nrho x 3), R (3x3 or list of 3x3))."""
    if np.size(ang0) != 3:
        raise ValueError('First input (initial frame) must contain three numbers, the three Euler angles.')
    if isinstance(nRot, str):
        nRot = F.letter2vec(nRot)
    elif np.size(nRot) != 3:
        raise ValueError("Second input (nRot) must be either a 3-element vector or a letter indicating "
                         "the vector ('x', 'y', etc).")
    rho = np.asarray(rho, dtype=float).ravel(order='F')
    nrho = rho.size
    ang = np.zeros((nrho, 3))
    R = [None] * nrho
    R0 = np.asarray(F.erot(ang0))
    for irho in range(nrho):
        Rrot = np.asarray(F.rotaxi2mat(nRot, float(rho[irho])))
        R[irho] = R0 @ Rrot
        ang[irho, :] = np.asarray(F.eulang(R[irho], True)).ravel()
    if nrho == 1:
        R = R[0]
    return ang, R
