"""Real roots in [0, 1] of a cubic polynomial (port of EasySpin's cubicsolve mex, C source).

    r, d1, d2 = cubicsolve(C, LoopFields=True)

C = [C1 C2 C3 C4] for y = C1*t^3 + C2*t^2 + C3*t + C4. With LoopFields false at most one root is
returned. d1 and d2 are the first and second derivatives. As in the mex, only the roots are
sorted; d1 and d2 stay in solver order. One deviation: for a root at t = 0 (C4 == 0) the mex
leaves d1/d2 uninitialised, and the port returns the actual derivatives there (DEVIATIONS.md).
"""
import math

import numpy as np
from numba import njit

from .._registry import nout

_PI = 3.141592653589793238462643383279502884
_EPS = 1e-11
_MAX_NEWTON = 11


@njit(cache=True, error_model='numpy')
def _exact_solver(C1, C2, C3, C4, S):
    """All real roots 0 <= r <= 1 by the exact formulas; -1 if all coefficients are zero."""
    N = 0
    if C1 != 0:
        a = C2 / C1
        b = C3 / C1
        c = C4 / C1
        Q = (a * a - 3 * b) / 9
        R = (2 * a * a * a - 9 * a * b + 27 * c) / 54
        QQQ = Q * Q * Q
        if R * R < QQQ:
            theta = math.acos(R / math.sqrt(QQQ))
            Q2 = -2 * math.sqrt(Q)
            s1 = Q2 * math.cos(theta / 3) - a / 3
            s2 = Q2 * math.cos((theta + 2 * _PI) / 3) - a / 3
            s3 = Q2 * math.cos((theta - 2 * _PI) / 3) - a / 3
            if s1 >= 0 and s1 <= 1:
                S[N] = s1
                N += 1
            if s2 >= 0 and s2 <= 1:
                S[N] = s2
                N += 1
            if s3 >= 0 and s3 <= 1:
                S[N] = s3
                N += 1
        else:
            A = -math.pow(abs(R) + math.sqrt(R * R - QQQ), 1 / 3.0)
            if R < 0:
                A = -A
            B = 0.0 if A == 0 else Q / A
            s1 = A + B - a / 3
            if s1 >= 0 and s1 <= 1:
                S[N] = s1
                N += 1
    elif C2 != 0:
        R = C3 * C3 - 4 * C2 * C4
        if R >= 0:
            Q = -(C3 + (-1.0 if C3 < 0 else 1.0) * math.sqrt(R)) / 2
            s1 = Q / C2
            s2 = C4 / Q
            if s1 >= 0 and s1 <= 1:
                S[N] = s1
                N += 1
            if s2 >= 0 and s2 <= 1:
                S[N] = s2
                N += 1
    elif C3 != 0:
        s1 = -C4 / C3
        if s1 >= 0 and s1 <= 1:
            S[N] = s1
            N += 1
    elif C4 != 0:
        pass
    else:
        N = -1
    return N


@njit(cache=True, error_model='numpy')
def _cubicsolve(a, b, c, d, single_root, S):
    is_monotonic = False
    if single_root and c * (3.0 * a + 2.0 * b + c) > 0:
        if c * a < 0:
            is_monotonic = True
    same_sign = (a + b + c + d) * d > 0
    if is_monotonic:
        if same_sign:
            return 0
    else:
        yex = d + c * (2 * a + b) / (3 * a + 2 * b)
        if same_sign and yex * d > 0:
            return 0
    if d == 0:
        S[0] = 0.0
        S[3] = c            # mex: uninitialised; here the derivatives at t = 0
        S[6] = 2 * b
        return 1
    n = 0
    if is_monotonic:
        t = -d / (a + b + c)
        for _ in range(_MAX_NEWTON):
            delta = (((a * t + b) * t + c) * t + d) / ((3 * a * t + 2 * b) * t + c)
            t -= delta
            if t < 0 or t > 1:
                break
            if abs(delta) < _EPS:
                n = 1
                S[0] = t
                break
    if n == 0:
        n = _exact_solver(a, b, c, d, S)
        if n < 0:
            return n
    for i in range(n):
        t = S[i]
        S[i + 3] = (3 * a * t + 2 * b) * t + c
        S[i + 6] = 6 * a * t + 2 * b
    return n


@nout(3)
def cubicsolve(C, LoopFields=None):
    """(r, d1, d2): roots in [0, 1] (sorted) and the derivatives in solver order (as the mex)."""
    C = np.asarray(C, dtype=float).ravel()
    if LoopFields is None:
        loop = True
    else:
        lf = np.asarray(LoopFields).ravel()
        if lf.size != 1:
            raise ValueError('LoopFields must be a scalar (boolean)!')
        loop = bool(lf[0] != 0)
    S = np.zeros(9)
    n = _cubicsolve(C[0], C[1], C[2], C[3], not loop, S)
    if n < 0:
        raise ValueError('All coefficients of the cubic spline are zero!')
    roots = S[:n].copy()
    for i in range(n):                 # the mex's insertion sort, applied to the roots only
        t = roots[i]
        j = i
        while j > 0 and roots[j - 1] > t:
            roots[j] = roots[j - 1]
            j -= 1
        roots[j] = t
    return roots, S[3:3 + n].copy(), S[6:6 + n].copy()
