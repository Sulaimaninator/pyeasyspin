"""Wigner 3-j symbol"""
from fractions import Fraction
import math
import numpy as np
from scipy.special import gammaln


def _pm1(x):
    """MATLAB (-1)^x: exact +-1 for integer x, complex exp(i*pi*x) otherwise."""
    x = float(x)
    if x == math.floor(x):
        return -1.0 if int(x) % 2 else 1.0
    return complex(-1) ** x


def _facln(x):
    return gammaln(x + 1)


def _binoln(n, k):
    return _facln(n) - _facln(k) - _facln(n - k)


def _binom_bi(n, k):
    v = 1
    for q in range(int(n - k + 1), int(n) + 1):
        v *= q
    for q in range(1, int(k) + 1):
        v //= q   # BigInteger.divide truncates (division is exact here)
    return v


def _isint(x):
    return x == math.floor(x)


def wigner3j(*varargin):
    """wigner3j(j1,j2,j3,m1,m2,m3[, method]) or ([j1 m1],[j2 m2],[j3 m3]), ([j1 j2 j3],[m1 m2 m3]),
    (2x3 array). Methods: 'f', 'b' with optional '+' (fast special cases); default 'f+' ('b+' if j>20)."""
    if varargin and isinstance(varargin[-1], str):
        Method = varargin[-1]
        inputs = varargin[:-1]
    else:
        Method = ''
        inputs = varargin
    fl = lambda a: np.asarray(a, dtype=float).ravel(order='F')
    if len(inputs) == 6:
        j1, j2, j3, m1, m2, m3 = (float(fl(a)[0]) for a in inputs)
    elif len(inputs) == 3:
        a, b, c = (fl(x) for x in inputs)
        j1, m1, j2, m2, j3, m3 = a[0], a[1], b[0], b[1], c[0], c[1]
    elif len(inputs) == 2:
        j, m = fl(inputs[0]), fl(inputs[1])
        j1, j2, j3, m1, m2, m3 = j[0], j[1], j[2], m[0], m[1], m[2]
    elif len(inputs) == 1:
        jm = np.asarray(inputs[0], dtype=float)
        if jm.shape != (2, 3):
            raise ValueError('If all J and M are supplied in an array, the array must be 2x3.')
        j1, j2, j3 = jm[0]
        m1, m2, m3 = jm[1]
    else:
        raise ValueError('Wrong number of parameters!')
    j1, j2, j3, m1, m2, m3 = (float(x) for x in (j1, j2, j3, m1, m2, m3))
    if not Method:
        Method = 'f+'
        if max(j1, j2, j3) > 20:
            Method = 'b+'
    jjjmmm = [j1, j2, j3, m1, m2, m3]
    if any(not _isint(2 * x) for x in jjjmmm):
        raise ValueError('Nonphysical parameters. All parameters must be integers or half-integers.')
    if j1 < 0:
        raise ValueError('Nonphysical parameter. j1 must satisfy j1>=0.')
    if j2 < 0:
        raise ValueError('Nonphysical parameter. j2 must satisfy j2>=0.')
    if j3 < 0:
        raise ValueError('Nonphysical parameter. j3 must satisfy j3>=0.')
    if not _isint(j1 - m1):
        raise ValueError('Nonphysical parameter. m1 must be one of -j1,-j1+1,...,j1-1,j1.')
    if not _isint(j2 - m2):
        raise ValueError('Nonphysical parameter. m2 must be one of -j2,-j2+1,...,j2-1,j2.')
    if not _isint(j3 - m3):
        raise ValueError('Nonphysical parameter. m3 must be one of -j3,-j3+1,...,j3-1,j3.')
    if m1 + m2 + m3 != 0:
        return 0.0
    if not ((j1 + j2 >= j3) and (j2 + j3 >= j1) and (j3 + j1 >= j2)):
        return 0.0
    if m1 == 0 and m2 == 0 and m3 == 0 and ((j1 + j2 + j3) % 2):
        return 0.0
    if all(x == 0 for x in jjjmmm):
        return 1.0
    useFast = '+' in Method
    if useFast and (j1 <= 2 or j2 <= 2 or j3 <= 2):
        return _fastwigner(j1, j2, j3, m1, m2, m3)
    if useFast and m1 == 0 and m2 == 0 and m3 == 0:
        J = j1 + j2 + j3
        if J < 100000:
            C, B, A = sorted([-j1 + j2 + j3, j1 - j2 + j3, j1 + j2 - j3])
            v = 1 / (J + 1)
            for i in range(1, int(B / 2) + 1):
                v = v / i * (B / 2 + i) * (A / 2 + i) ** 2 / (A + B / 2 + i) / (A + i)
            for i in range(1, int(C / 2) + 1):
                v = v / i * (C / 2 + i) * (A / 2 + B / 2 + i) ** 2 / (A + B + C / 2 + i) / (A + B + i)
            return _pm1(J / 2) * np.sqrt(v)
    tmin = max(0, j1 - j3 + m2, j2 - j3 - m1)
    tmax = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    pre = (_facln(j1 + m1) + _facln(j1 - m1) + _facln(j2 + m2) + _facln(j2 - m2)
           + _facln(j3 + m3) + _facln(j3 - m3) - _facln(j1 + j2 + j3 + 1)
           - _facln(j1 + j2 - j3) - _facln(j1 - j2 + j3) - _facln(-j1 + j2 + j3))
    ts = np.arange(tmin, tmax + 0.5)
    if 'f' in Method:
        binsum = 0
        for t in ts:
            p = _binoln(j1 + j2 - j3, t) + _binoln(j1 - j2 + j3, j1 - m1 - t) + _binoln(-j1 + j2 + j3, j2 + m2 - t)
            p = _pm1(t) * np.exp(p + pre / 2)
            binsum = binsum + p
        return _pm1(j1 - j2 - m3) * binsum
    elif 'b' in Method:
        t = tmin
        p = _binom_bi(j1 + j2 - j3, t)
        p = p * _binom_bi(j1 - j2 + j3, j1 - m1 - t)
        p = p * _binom_bi(-j1 + j2 + j3, j2 + m2 - t)
        p = p * int(_pm1(t))
        binsum = p
        for t in ts[1:]:
            q1 = int(round((j1 + j2 - j3 - t + 1) * (j1 - m1 - t + 1) * (j2 + m2 - t + 1)))
            q2 = int(round(t * (-j2 + j3 + m1 + t) * (-j1 + j3 - m2 + t)))
            p = p * q1
            p = (abs(p) // abs(q2)) * (1 if (p >= 0) == (q2 > 0) else -1)   # BigInteger.divide truncates
            p = -p
            binsum = binsum + p
        n = len(str(binsum)) - 1
        b = float(Fraction(binsum, 10 ** n))
        return _pm1(j1 - j2 - m3) * np.exp(pre / 2 + n * np.log(10)) * b
    raise ValueError('Unknown computation method.')


def _fastwigner(j1, j2, j3, m1, m2, m3):
    phase = 1
    parity = _pm1(j1 + j2 + j3)
    if j1 <= 2:
        j1, j3 = j3, j1
        m1, m3 = m3, m1
        phase = phase * parity
    elif j2 <= 2:
        j2, j3 = j3, j2
        m2, m3 = m3, m2
        phase = phase * parity
    elif j3 <= 2:
        pass
    else:
        raise ValueError('At least one J must be <=2.')
    if m3 < 0:
        m1, m2, m3 = -m1, -m2, -m3
        phase = phase * parity
    if j1 < j2:
        j1, j2 = j2, j1
        m1, m2 = m2, m1
        phase = phase * parity
    J = j2
    M = m1
    JmM = J - M
    JpM = J + M
    jdelta = j1 - j2
    x = j1 + j2 - 1
    phase = phase * _pm1(J - M)
    sq = np.sqrt
    if j3 == 0:
        val = 1 / sq(2 * j1 + 1)
    elif j3 == 2:
        tmp2 = x * (x + 1) * (x + 2) * (x + 3) * (x + 4)
        if m3 == 0:
            if jdelta == 0:
                val = 2 * (3 * M ** 2 - J * (J + 1)) / sq(tmp2)
            elif jdelta == 1:
                tmp1 = 6 * (JpM + 1) * (JmM + 1)
                val = -2 * M * sq(tmp1 / tmp2)
            else:
                tmp1 = 6 * (JpM + 2) * (JpM + 1) * (JmM + 2) * (JmM + 1)
                val = sq(tmp1 / tmp2)
        elif m3 == 1:
            if jdelta == 0:
                tmp1 = 6 * (JpM + 1) * JmM
                val = (1 + 2 * M) * sq(tmp1 / tmp2)
            elif jdelta == 1:
                tmp1 = (JmM + 1) * JmM
                val = -2 * (J + 2 * M + 2) * sq(tmp1 / tmp2)
            else:
                tmp1 = (JpM + 2) * (JmM + 2) * (JmM + 1) * JmM
                val = 2 * sq(tmp1 / tmp2)
        else:
            if jdelta == 0:
                tmp1 = 6 * (JmM - 1) * JmM * (JpM + 1) * (JpM + 2)
                val = sq(tmp1 / tmp2)
            elif jdelta == 1:
                tmp1 = (JmM - 1) * JmM * (JmM + 1) * (JpM + 2)
                val = -2 * sq(tmp1 / tmp2)
            else:
                tmp1 = (JmM - 1) * JmM * (JmM + 1) * (JmM + 2)
                val = sq(tmp1 / tmp2)
    elif j3 == 0.5:
        val = -1j * sq((JmM + 0.5) / (2 * J + 2) / (2 * J + 1))
    elif j3 == 1:
        if m3 == 0 and jdelta == 0:
            tmp2 = J * (J + 1) * (x + 2)
        else:
            tmp2 = (x + 1) * (x + 2) * (x + 3)
        if m3 == 0:
            if jdelta == 0:
                val = M / sq(tmp2)
            else:
                tmp1 = (JmM + 1) * (JpM + 1) * 2
                val = -sq(tmp1 / tmp2)
        else:
            if jdelta == 0:
                tmp1 = JmM * (JpM + 1) * 2
                val = sq(tmp1 / tmp2)
            else:
                tmp1 = JmM * (JmM + 1)
                val = -sq(tmp1 / tmp2)
    elif j3 == 1.5:
        if m3 == 1.5:
            if jdelta == 1.5:
                val = 1j * sq((JmM - .5) * (JmM + .5) * (JmM + 1.5) / (2 * J + 4) / (2 * J + 3) / (2 * J + 2) / (2 * J + 1))
            else:
                val = -1j * sq(3 * (JmM - .5) * (JmM + .5) * (JpM + 1.5) / (2 * J + 3) / (2 * J + 2) / (2 * J + 1) / (2 * J))
        else:
            if jdelta == 1.5:
                val = 1j * sq(3 * (JmM + .5) * (JmM + 1.5) * (JpM + 1.5) / (2 * J + 4) / (2 * J + 3) / (2 * J + 2) / (2 * J + 1))
            else:
                val = -1j * sq((JmM + .5) / (2 * J + 3) / (2 * J + 2) / (2 * J + 1) / (2 * J)) * (J + 3 * M + 1.5)
    else:
        raise ValueError('j3 must be 0, 1/2, 1, 3/2, or 2.')
    return val * phase
