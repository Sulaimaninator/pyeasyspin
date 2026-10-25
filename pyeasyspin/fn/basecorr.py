"""Baseline correction"""
import numpy as np
from .._registry import nout
from .._mat import isnumeric, isempty, vec


def _lsq(D, b):
    return np.linalg.lstsq(D, b, rcond=None)[0]


@nout(2)
def basecorr(data=None, dim=None, n=None, region=None):
    """basecorr(data, dim, n, region=None) -> (datacorr, baseline). dim = 1 (columns), 2 (rows) or
    [] (2-D fit, n = [n1 n2]); polynomial order n <= 6; region = logical mask of points to fit."""
    if data is None:
        raise ValueError('Provide dimension dim (second input) and polynomial order n (third input).')
    if n is None:
        raise ValueError('Provide polynomial order n (third input).')
    maxOrder = 6
    if not isnumeric(data) or isempty(data):
        raise ValueError('Data must be a non-empty numerical array.')
    data = np.asarray(data)
    if data.ndim < 2:
        data = data.reshape(1, -1)
    if data.ndim > 2:
        raise ValueError('Only 1D or 2D data can be fitted. You provided a %dD data array.' % data.ndim)
    twoDimFit = isempty(dim)
    if not twoDimFit:
        dim = int(np.ravel(dim)[0])
        if dim not in (1, 2):
            raise ValueError('dim (2nd input) must either 1 (fit columns), 2 (fit rows), or [] (2D fit).')
    rowVector = False
    if not twoDimFit and dim == 1 and data.shape[0] == 1:
        rowVector = True
        data = vec(data).reshape(-1, 1)
    nv = vec(np.asarray(n, dtype=float))
    if np.any(nv < 0) or np.any(np.mod(nv, 1)) or np.any(nv > maxOrder):
        raise ValueError('Order must contain integers between 0 and %d!' % maxOrder)
    if twoDimFit:
        if nv.size != 2:
            raise ValueError('For 2D fit, the polynomial order n (3rd input) must have 2 elements, one for each dimension.')
        for i in range(2):
            if nv[i] >= data.shape[i]:
                raise ValueError('The provided polynomial order n=%d must be smaller than the number of points, %d.'
                                 % (nv[i], data.shape[i]))
    else:
        if nv.size != 1:
            raise ValueError('Polynomial order (3rd input) must be a single number.')
        if nv[0] >= data.shape[dim - 1]:
            raise ValueError('The provided polynomial order n=%d must be smaller than the number of points, %d.'
                             % (nv[0], data.shape[dim - 1]))
    if not isempty(region):
        region = np.asarray(region)
        if region.dtype != bool:
            raise ValueError('region (4th input) must be a logical array.')
    if twoDimFit:
        r, c = data.shape
        x = np.tile(np.linspace(-1, 1, r), c)                 # reshape(repmat(...).',[],1)
        y = np.repeat(np.linspace(-1, 1, c), r)
        n1, n2 = int(nv[0]), int(nv[1])
        D = np.zeros((r * c, (n1 + 1) * (n2 + 1)))
        col = 0
        for j in range(n2 + 1):
            for i in range(n1 + 1):
                D[:, col] = x ** i * y ** j
                col += 1
        p = _lsq(D, vec(data))
        baseline = (D @ p).reshape(r, c, order='F')
    else:
        if dim == 2:
            data = data.T
        x = np.linspace(-1, 1, data.shape[0])
        D = x.reshape(-1, 1) ** np.arange(int(nv[0]) + 1)
        if isempty(region):
            p = _lsq(D, data)
        else:
            m = vec(region)
            p = _lsq(D[m, :], data[m, :])
        baseline = D @ p
        if dim == 2:
            baseline = baseline.T
            data = data.T
    datacorr = data - baseline
    if rowVector:
        datacorr = datacorr.T
        baseline = baseline.T
    return datacorr, baseline
