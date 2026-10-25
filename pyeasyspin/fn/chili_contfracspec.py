"""Compute spectrum using continued-fraction method"""
import numpy as np

from .._registry import nout


def _v(x):
    a = np.asarray(x)
    return a.ravel(order='F') if a.ndim else a.reshape(1)


@nout(1)
def chili_contfracspec(z=None, alpha=None, beta=None, k=None, useLentz=None):
    """chili_contfracspec(z, alpha, beta, k=numel(alpha), useLentz=False) -> spec: the continued
    fraction 1/(z+alpha(1)-beta(1)^2/(z+alpha(2)-...)) truncated after k levels, evaluated at
    every z (right-to-left, or the modified Lentz recursion when useLentz is true). spec has the
    shape of z."""
    zz = np.asarray(z)
    alpha = _v(alpha)
    beta = _v(beta)
    if k is None:
        k = alpha.size
    k = int(k)
    if alpha.size < k or beta.size < k:
        raise ValueError('alpha and/or beta have less than than %d elements!' % k)
    if useLentz is None:
        useLentz = False
    if useLentz:
        tiny = 1e-30
        spec = tiny
        C = spec
        D = 0
        for q in range(k):
            Cold = C
            Dold = D
            specold = spec
            b = zz + alpha[q]
            a = 1 if q == 0 else -(beta[q - 1] * beta[q - 1])
            C = b + a / Cold
            D = b + a * Dold
            D = 1 / D
            Delta = C * D
            spec = specold * Delta
    else:
        spec = np.inf
        with np.errstate(divide='ignore', invalid='ignore'):
            for m in range(k - 1, -1, -1):
                spec = zz + alpha[m] - (beta[m] * beta[m]) / spec
        spec = 1 / spec
    return spec
