"""Compute Wigner D-matrix element from a quaternion"""
import math
import numpy as np


def _isint(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and float(v) % 1.0 == 0


def stochtraj_wignerdquat(L=None, M=None, K=None, q=None):
    """stochtraj_wignerdquat(L, M, K, q) -> Del (1 x N complex), the Wigner D-matrix element D^L_MK of
    the rotations given by the quaternions q (4 x ..., N = numel(q)/4). EasySpin's persistent cache
    of factorials is recomputed on every call (no effect on the values)."""
    def num1(v):
        a = np.asarray(v)
        return a.size == 1 and np.isrealobj(a) and np.issubdtype(a.dtype, np.number)
    if not num1(L) or float(np.asarray(L).ravel()[0]) % 1.0 or float(np.asarray(L).ravel()[0]) < 0:
        raise ValueError('Index L must be an integer greater than zero.')
    L = float(np.asarray(L).ravel()[0])
    if not num1(M) or float(np.asarray(M).ravel()[0]) % 1.0 or abs(float(np.asarray(M).ravel()[0])) > L:
        raise ValueError('Index M must be in the range -L<=M<=L.')
    M = float(np.asarray(M).ravel()[0])
    if not num1(K) or float(np.asarray(K).ravel()[0]) % 1.0 or abs(float(np.asarray(K).ravel()[0])) > L:
        raise ValueError('Index K must be in the range -L<=K<=L.')
    K = float(np.asarray(K).ravel()[0])
    q = np.asarray(q)
    if not np.issubdtype(q.dtype, np.number) or q.ndim < 1 or q.shape[0] != 4:
        raise ValueError('q must be an array of size (4,...) array.')
    q = q.reshape(4, -1, order='F')
    fact = lambda n: float(math.factorial(int(n)))
    prefactor = math.sqrt(fact(L + M) * fact(L - M) * fact(L + K) * fact(L - K))
    s = np.arange(max(0.0, K - M), min(L + K, L - M) + 0.5)
    powers = np.array([L + K - s, M - K + s, s, L - M - s])               # 4 x ns
    denoms = np.vectorize(fact, otypes=[float])(powers) if s.size else np.zeros((4, 0))
    Q = np.array([q[0] - 1j * q[3],
                  -1j * q[1] - q[2],
                  -1j * q[1] + q[2],
                  q[0] + 1j * q[3]])                                        # 4 x N
    N = q.shape[1]
    total = np.zeros(N, dtype=complex)
    for k in range(s.size):
        t = (Q[0] ** powers[0, k]) / denoms[0, k]
        for r in range(1, 4):
            t = t * ((Q[r] ** powers[r, k]) / denoms[r, k])
        total = total + t
    return (prefactor * total).reshape(1, N)
