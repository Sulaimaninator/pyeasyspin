"""Matrix exponential, using multiple algorithms"""
import math
import numpy as np
import scipy.linalg
import scipy.sparse.linalg


def _dense(A):
    return A.toarray() if hasattr(A, 'toarray') else np.asarray(A)


def _ishermitian(A):
    return A.ndim == 2 and A.shape[0] == A.shape[1] and np.array_equal(A, A.conj().T)


def expm_(A=None, algo=None):
    """expm_(A, algo): 'orig' (default, MATLAB expm -> scipy.linalg.expm), 'fast' (EasySpin's
    trimmed Higham scaling-and-squaring) or 'eig' (V*diag(exp(d))*V')."""
    if algo is None:
        algo = 'orig'
    if algo == 'orig':
        if hasattr(A, 'toarray'):
            return scipy.sparse.linalg.expm(A.tocsc())
        return scipy.linalg.expm(np.asarray(A))
    if algo == 'fast':
        return _expm_fast1(_dense(A))
    if algo == 'eig':
        A = _dense(A)
        if _ishermitian(A):            # MATLAB eig uses the Hermitian solver then
            D, V = scipy.linalg.eigh(A)
        else:
            D, V = np.linalg.eig(A)
        return V @ np.diag(np.exp(D)) @ V.conj().T
    return None


def _norm1(A):
    return float(np.max(np.sum(np.abs(A), axis=0))) if A.size else 0.0


def _expm_fast1(A):
    T = A
    if np.count_nonzero(T - np.diag(np.diag(T))) == 0:
        return np.diag(np.exp(np.diag(T)))
    s, m, T2, T4, T6 = _expm_params(T)
    if s != 0:
        T = T / (2.0 ** s)
        T2 = T2 / 2.0 ** (s * 2)
        T4 = T4 / 2.0 ** (s * 4)
        T6 = T6 / 2.0 ** (s * 6)
    c = {3: [120, 60, 12, 1],
         5: [30240, 15120, 3360, 420, 30, 1],
         7: [17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1],
         9: [17643225600, 8821612800, 2075673600, 302702400, 30270240,
             2162160, 110880, 3960, 90, 1],
         13: [64764752532480000, 32382376266240000, 7771770303897600,
              1187353796428800, 129060195264000, 10559470521600,
              670442572800, 33522128640, 1323241920,
              40840800, 960960, 16380, 182, 1]}[m]
    c = [float(x) for x in c]
    I = np.eye(T.shape[0])
    if m in (3, 5, 7, 9):
        Tp = {2: T2, 4: T4, 6: T6}       # 1-based cell indices of Tpowers
        for k in range(8, m, 2):
            Tp[k] = Tp[k - 2] @ Tp[2]
        U = c[1] * I
        V = c[0] * I
        for j in range(m, 2, -2):
            U = U + c[j] * Tp[j - 1]
            V = V + c[j - 1] * Tp[j - 1]
        U = T @ U
    else:
        U = T @ (T6 @ (c[13] * T6 + c[11] * T4 + c[9] * T2) + c[7] * T6 + c[5] * T4 + c[3] * T2 + c[1] * I)
        V = T6 @ (c[12] * T6 + c[10] * T4 + c[8] * T2) + c[6] * T6 + c[4] * T4 + c[2] * T2 + c[0] * I
    Fm = np.linalg.solve(V - U, 2 * U) + I
    for _ in range(int(s)):
        Fm = Fm @ Fm
    return Fm


def _normAm(A, m):
    """norm(A^m,1). For n >= 50 MATLAB estimates it with normest1 (randomized) unless A >= 0;
    here the exact value is used (only affects the choice of the scaling power)."""
    n = A.shape[0]
    if n >= 50 and np.isrealobj(A) and np.all(A >= 0):
        e = np.ones(n)
        for _ in range(m):
            e = A.T @ e
        return float(np.max(np.abs(e)))
    return _norm1(np.linalg.matrix_power(A, m))


def _ell(T, coeff, m_val):
    scaledT = coeff ** (1 / (2 * m_val + 1)) * np.abs(T)
    alpha = _normAm(scaledT, 2 * m_val + 1) / _norm1(T)
    with np.errstate(divide='ignore'):
        v = np.log2(2 * alpha / np.finfo(float).eps) / (2 * m_val) if alpha > 0 else -np.inf
    return max(math.ceil(v) if np.isfinite(v) else v, 0)


def _expm_params(T):
    coeff = [1 / 100800, 1 / 10059033600, 1 / 4487938430976000,
             1 / 5914384781877411840000, 1 / 113250775606021113483283660800000000]
    s = 0
    theta = [1.495585217958292e-002, 2.539398330063230e-001, 9.504178996162932e-001,
             2.097847961257068e+000, 5.371920351148152e+000]
    T2 = T @ T
    T4 = T2 @ T2
    T6 = T2 @ T4
    d4 = _norm1(T4) ** (1 / 4)
    d6 = _norm1(T6) ** (1 / 6)
    eta1 = max(d4, d6)
    if eta1 <= theta[0] and _ell(T, coeff[0], 3) == 0:
        return s, 3, T2, T4, T6
    if eta1 <= theta[1] and _ell(T, coeff[1], 5) == 0:
        return s, 5, T2, T4, T6
    isSmall = T.shape[0] < 150
    d8 = (_norm1(T4 @ T4) if isSmall else _normAm(T4, 2)) ** (1 / 8)
    eta3 = max(d6, d8)
    if eta3 <= theta[2] and _ell(T, coeff[2], 7) == 0:
        return s, 7, T2, T4, T6
    if eta3 <= theta[3] and _ell(T, coeff[3], 9) == 0:
        return s, 9, T2, T4, T6
    d10 = (_norm1(T4 @ T6) if isSmall else _normAm(T2, 5)) ** (1 / 10)
    eta4 = max(d8, d10)
    eta5 = min(eta3, eta4)
    s = max(math.ceil(np.log2(eta5 / theta[4])) if eta5 > 0 else 0, 0)
    s = s + _ell(T / 2 ** s, coeff[4], 13)
    if np.isinf(s):
        t, s = np.frexp(_norm1(T) / theta[-1])
        s = int(s) - (t == 0.5)
    return int(s), 13, T2, T4, T6
