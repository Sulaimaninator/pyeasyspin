"""'Wrapped' Gaussian function"""
import numpy as np
from .._registry import F


def wrappedgaussian(x=None, x0=None, fwhm=None, domain=None):
    """wrappedgaussian(x (row), x0, fwhm, domain=[-pi, pi]) -> y, normalized with trapz over x."""
    if domain is None:
        domain = np.array([[-np.pi, np.pi]])
    fwhm_a = np.asarray(fwhm)
    if np.any(fwhm_a <= 0) or np.iscomplexobj(fwhm_a):
        raise ValueError('fwhm must be positive and real!')
    if fwhm_a.size > 1:
        raise ValueError('fwhm must contain 1 element!')
    d = np.asarray(domain)
    if not (d.ndim == 1 or (d.ndim == 2 and d.shape[0] == 1)) or d.size != 2 or np.iscomplexobj(d):
        raise ValueError('domain must be a 2-element row vector of real numbers.')
    d = d.ravel()
    if d[0] >= d[1]:
        raise ValueError('domain elements must be given in increasing order.')
    if x0 > d[1] or x0 < d[0]:
        raise ValueError('x0 must be contained within the given domain values.')
    fwhm = float(fwhm_a.ravel()[0])
    shift = d[1] - d[0]
    sigma = fwhm / 2 / np.sqrt(2 * np.log(2))
    xmin = x0 - 5 * sigma
    xmax = x0 + 5 * sigma
    M = 2 * np.ceil((xmax - xmin) / shift / 2)
    k = np.arange(-M / 2, M / 2 + 1).reshape(-1, 1)
    x = np.asarray(x, dtype=float)
    xr = x.reshape(1, -1) if x.ndim < 2 else x
    xgrid = (xr - x0) + shift * k
    y = np.sum(F.gaussian(xgrid, 0, fwhm), axis=0, keepdims=True)
    y = y / np.trapezoid(y.ravel(), xr.ravel())
    return y
