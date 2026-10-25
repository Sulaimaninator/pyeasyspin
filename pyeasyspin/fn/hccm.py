"""Heteroscedasticity-consistent covariance matrix (sandwich estimator) (private)."""
import numpy as np

from .._registry import nout
from .._mat import isempty, size as msize, vec


@nout(1)
def hccm(J=None, res=None, mode=None):
    """hccm(J, res, mode) -> C. J: N x M Jacobian, res: residuals (a matrix is taken as the
    diagonal of V itself), mode 'HC0'..'HC5'."""
    J = np.asarray(J, dtype=float)
    if J.ndim < 2:
        J = J.reshape(-1, 1)
    N, M = J.shape
    if mode is None or isempty(mode) or all(s > 1 for s in msize(res)):
        diagV = np.asarray(res)
    else:
        diagV = None
    pinvJTJ = np.linalg.pinv(J.T @ J)
    h = np.sum((J @ pinvJTJ) * J, axis=1)
    if diagV is None:
        r = vec(np.asarray(res))
        m = str(mode).upper()
        if m == 'HC0':
            diagV = r**2
        elif m == 'HC1':
            diagV = N / (N - M) * r**2
        elif m == 'HC2':
            diagV = r**2 / (1 - h)
        elif m == 'HC3':
            diagV = (r / (1 - h))**2
        elif m == 'HC4':
            delta = np.minimum(4, N * h / M)
            diagV = r**2 / ((1 - h)**delta)
        elif m == 'HC5':
            M_ = 0.7
            alpha = np.minimum(max(4, M_ * np.max(h) / np.mean(h)), h / np.mean(h))
            diagV = r**2 / (np.sqrt((1 - h)**alpha))
        else:
            raise ValueError('HCCM estimation mode not found.')
    return ((pinvJTJ @ J.T) * vec(diagV)[None, :]) @ J @ pinvJTJ
