"""Calculate the torque due to an anisotropic orienting potential."""
import math
import numpy as np
from .._registry import F


def _Cpm(L, K, pm):
    if pm == '+':
        return math.sqrt(L * (L + 1) - K * (K + 1))
    return math.sqrt(L * (L + 1) - K * (K - 1))


def _iJu(L, M, K, lam, q):
    if K == L:
        iJxu = 1j / 2 * lam * _Cpm(L, K, '-') * F.stochtraj_wignerdquat(L, M, K - 1, q)
        iJyu = -1 / 2 * lam * _Cpm(L, K, '-') * F.stochtraj_wignerdquat(L, M, K - 1, q)
    elif K == -L:
        iJxu = 1j / 2 * lam * _Cpm(L, K, '+') * F.stochtraj_wignerdquat(L, M, K + 1, q)
        iJyu = 1 / 2 * lam * _Cpm(L, K, '+') * F.stochtraj_wignerdquat(L, M, K + 1, q)
    else:
        iJxu = 1j / 2 * lam * (_Cpm(L, K, '+') * F.stochtraj_wignerdquat(L, M, K + 1, q)
                               + _Cpm(L, K, '-') * F.stochtraj_wignerdquat(L, M, K - 1, q))
        iJyu = 1 / 2 * lam * (_Cpm(L, K, '+') * F.stochtraj_wignerdquat(L, M, K + 1, q)
                              - _Cpm(L, K, '-') * F.stochtraj_wignerdquat(L, M, K - 1, q))
    iJzu = 1j * lam * K * F.stochtraj_wignerdquat(L, M, K, q)
    return np.concatenate([np.atleast_2d(iJxu), np.atleast_2d(iJyu), np.atleast_2d(iJzu)], axis=0)


def stochtraj_calcanistorque(LMK=None, lambda_=None, q=None):
    """stochtraj_calcanistorque(LMK, lambda, q) -> torque (3 x nTraj): torque of the orienting potential
    sum_n lambda_n (D^L_MK + c.c.) for the quaternions q (4 x nTraj). LMK is N x 3, lambda N x 1."""
    q = np.asarray(q)
    if not np.issubdtype(q.dtype, np.number) or q.shape[0] != 4:
        raise ValueError('q must be an array of size 4x....')
    LMK = np.atleast_2d(np.asarray(LMK))
    if LMK.ndim != 2 or LMK.shape[1] != 3:
        raise ValueError('LMK must be an array of shape Nx3.')
    lam = np.asarray(lambda_)
    lam = lam.reshape(1, 1) if lam.ndim == 0 else (lam.reshape(1, -1) if lam.ndim == 1 else lam)
    if lam.ndim != 2 or lam.shape[1] != 1:
        raise ValueError('lambda must be an array of shape Nx1.')
    nCoefs = lam.shape[0]
    LMKr = np.real(LMK).astype(float)
    N = q.shape[1] if q.ndim > 1 else 1
    iJvecu = np.zeros((3, N, nCoefs))
    for n in range(nCoefs):
        L, M, K = (float(v) for v in LMKr[n])
        lamn = lam[n, 0]
        lamn = lamn.item() if hasattr(lamn, 'item') else lamn
        if M == 0 and K == 0:
            iJvecu[:, :, n] = np.real(_iJu(L, M, K, lamn, q))
        else:
            iJvecu[:, :, n] = 2 * np.real(_iJu(L, M, K, lamn, q))
    if nCoefs > 1:
        torque = -np.sum(iJvecu, axis=2)
    else:
        torque = -iJvecu[:, :, 0]
    return torque
