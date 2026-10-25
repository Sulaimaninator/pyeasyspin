"""Wigner rotation matrices"""
import numpy as np
import scipy.linalg
from .._wignerd_tab import TAB


def _jacobip(n, a, b, x):
    P0 = 1
    if n == 0:
        return P0
    P1 = (a - b) / 2 + (1 + (a + b) / 2) * x
    if n == 1:
        return P1
    a2b2 = a ** 2 - b ** 2
    ab = a + b
    P = P1
    for N in range(2, int(n) + 1):
        P = ((2 * N + ab - 1) * ((2 * N + ab) * (2 * N + ab - 2) * x + a2b2) * P1
             - 2 * (N + a - 1) * (N + b - 1) * (2 * N + ab) * P0) / (2 * N * (N + ab) * (2 * N + ab - 2))
        P0 = P1
        P1 = P
    return P


def _mrange(a, b):
    """MATLAB a:-1:b (descending, unit step)."""
    n = int(np.floor(a - b + 1e-9)) + 1
    return a - np.arange(max(n, 0), dtype=float)


def _littled_jacobi(J, m1, m2, beta):
    beta = -beta
    cand = [J + m2, J - m2, J + m1, J - m1]
    idx = int(np.argmin(cand))
    k = cand[idx]
    if idx == 0:
        mu, nu, lam = m1 - m2, -m1 - m2, m1 - m2
    elif idx == 1:
        mu, nu, lam = m2 - m1, m1 + m2, 0
    elif idx == 2:
        mu, nu, lam = m2 - m1, -m1 - m2, 0
    else:
        mu, nu, lam = m1 - m2, m1 + m2, m1 - m2
    q_enum = np.concatenate([_mrange(2 * J - k, 2 * J - 2 * k - mu + 1), np.arange(1, nu + 1, dtype=float)])
    q_denom = np.concatenate([np.arange(1, k + mu + 1, dtype=float), np.arange(k + 1, k + nu + 1, dtype=float)])
    if J > 28:
        q_enum = np.sort(q_enum)
        q_denom = np.sort(q_denom)
    q = np.prod(q_enum / q_denom)
    return (-1) ** int(round(lam)) * np.sqrt(q) * np.sin(beta / 2) ** mu * np.cos(beta / 2) ** nu \
        * _jacobip(k, mu, nu, np.cos(beta))


def _littled_explicit(J, m1, m2, beta):
    beta = -beta
    if J == 0:
        return np.ones(np.shape(beta)) if np.ndim(beta) else 1.0
    ridx = J + 1 - m1
    cidx = J + 1 - m2
    linidx = int(round((cidx - 1) * (2 * J + 1) + ridx))
    return TAB[(float(J), linidx)](beta)


def _num(x):
    a = np.asarray(x, dtype=float)
    return float(a.ravel()[0]) if a.size == 1 else a


def wignerd(J=None, *varargin):
    """wignerd(J, beta[, phase]) or wignerd(J, alpha, beta, gamma[, phase]) -> D matrix (2J+1 square,
    rows/cols m = J..-J); J = [J m1 m2] gives one element. phase '-' (default) or '+'."""
    nargin = 1 + len(varargin)
    defaultphase = '-'
    alpha = gamma = None
    if nargin == 1:
        raise ValueError('Angles are missing')
    elif nargin == 2:
        beta, betaonly, phase = varargin[0], True, defaultphase
    elif nargin == 3:
        beta, betaonly, phase = varargin[0], True, varargin[1]
    elif nargin == 4:
        alpha, beta, gamma = varargin[:3]
        phase, betaonly = defaultphase, False
    elif nargin == 5:
        alpha, beta, gamma = varargin[:3]
        betaonly, phase = False, varargin[3]
    else:
        raise ValueError('Too many input arguments.')
    Ja = np.asarray(J, dtype=float).ravel(order='F')
    if Ja.size == 1:
        calculateMatrix = True
        J = float(Ja[0])
    elif Ja.size == 3:
        calculateMatrix = False
        m1, m2, J = float(Ja[1]), float(Ja[2]), float(Ja[0])
    else:
        raise ValueError('First input must be either J or [J m1 m2].')
    if (J % 0.5) or J < 0:
        raise ValueError('J must be one of 0, 1/2, 1, 3/2, 2, etc.')
    if not calculateMatrix:
        if m1 < -J or m1 > J:
            raise ValueError(f'm1 must be an integer between -J and J. Your J is {J:g}.')
        if m2 < -J or m2 > J:
            raise ValueError(f'm1 must be an integer between -J and J. Your J is {J:g}.')
    if not isinstance(phase, str) or len(phase) != 1:
        raise ValueError("Last argument must be either '+' or '-'.")
    if phase == '-':
        phase = -1
    elif phase == '+':
        phase = +1
    else:
        raise ValueError("Phase must be either '-' or '+'.")
    beta = _num(beta)
    if not betaonly:
        alpha = _num(alpha)
        gamma = _num(gamma)
    if calculateMatrix:
        n = int(round(2 * J + 1))
        if not np.any(beta) and (betaonly or (not np.any(alpha) and not np.any(gamma))):
            return np.eye(n)
        k = np.arange(1, n, dtype=float)
        v = np.sqrt(k * (2 * J + 1 - k)) / 2
        Jy = np.diag(v, 1) - np.diag(v, -1)
        d = scipy.linalg.expm(phase * beta * Jy)
        if betaonly:
            return d
        mz = J - np.arange(n)
        return np.outer(np.exp(1j * phase * alpha * mz), np.exp(1j * phase * gamma * mz)) * d
    if J <= 4:
        d = _littled_explicit(J, m1, m2, phase * beta)
    else:
        d = _littled_jacobi(J, m1, m2, phase * beta)
    if betaonly:
        return d
    return np.exp(phase * 1j * (alpha * m1 + gamma * m2)) * d
