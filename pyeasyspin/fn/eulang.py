"""Euler angles from rotation matrix"""
import numpy as np
from .._registry import nout


@nout(-1)
def eulang(R=None, nocheck=None, nargout=None):
    """eulang(R[, nocheck]) -> [alpha beta gamma] (1x3); nargout=3 -> (alpha, beta, gamma)."""
    if nocheck is None:
        nocheck = False
    R = np.asarray(R)
    if not nocheck:
        if R.shape != (3, 3) or np.iscomplexobj(R):
            raise ValueError('eulang: Rotation matrix must be a real-valued 3x3 matrix.')
        R = R.astype(float)
        orthogonalityError = np.linalg.norm(R.T @ R - np.eye(3), 2)
        if orthogonalityError > 1e-2:
            raise ValueError(f'eulang: Rotation matrix is not orthogonal, deviation is {orthogonalityError:f}.')
        elif orthogonalityError > 1e-6:
            print(f'eulang: Rotation matrix is not orthogonal, deviation is {orthogonalityError:g}.')
            print('eulang: Orthogonalizing using singular-value decomposition (SVD).')
            U, _, Vh = np.linalg.svd(R)
            R = U @ Vh
        if np.linalg.det(R) < 0:
            raise ValueError('eulang: Rotation matrix has negative determinant. Change the signs in one column or row.')
    R = R.astype(float)
    lim = 1e-8
    if abs(R[2, 2] - 1) <= lim:
        alpha = np.arctan2(R[0, 1], R[1, 1]); beta = 0.0; gamma = 0.0
    elif abs(R[2, 2] + 1) <= lim:
        alpha = np.arctan2(-R[0, 1], R[1, 1]); beta = np.pi; gamma = 0.0
    else:
        alpha = np.arctan2(R[2, 1], R[2, 0])
        beta = np.arctan2(np.sqrt(R[2, 0] ** 2 + R[2, 1] ** 2), R[2, 2])
        gamma = np.arctan2(R[1, 2], -R[0, 2])
    thr = -1e-8
    if alpha < thr:
        alpha = alpha + 2 * np.pi
    if gamma < thr:
        gamma = gamma + 2 * np.pi
    if beta < 0:
        beta = -beta
        if alpha < 0:
            alpha += np.pi; gamma += np.pi
        else:
            alpha -= np.pi; gamma -= np.pi
    alpha, beta, gamma = float(alpha), float(beta), float(gamma)
    if nargout in (None, 0, 1):
        return np.array([alpha, beta, gamma])
    if nargout == 3:
        return alpha, beta, gamma
    raise ValueError('Wrong number of output arguments.')
