"""Exponentially weighted recursive least squares adaptive filter"""
import numpy as np


def _rls(data, d, p, lam, delta, nPreAvg):
    nPoints, nScans = data.shape
    y = np.zeros((nPoints, nScans))
    P = delta * np.eye(p)
    w = np.zeros(p)
    for iScan in range(nPreAvg, nScans):
        for k in range(p - 1, nPoints):
            x = data[k - p + 1:k + 1, iScan]
            phi = P @ x
            g = phi / (lam + x @ phi)
            y[k, iScan] = x @ w
            e = d[k] - y[k, iScan]
            w = w + g * e
            P = (P - np.outer(g, phi)) / lam
            P = (P + P.T) / 2
    y = y[:, nPreAvg:]
    return np.mean(y, axis=1)


def ewrls(data=None, p=None, lambda_=None, nPreAvg=None, delta=None, direc=None, plot=False):
    """ewrls(data, p, lambda, nPreAvg=0, delta=100, direc='fb') -> y (nPoints x 1): RLS-filtered
    average of the scans in the columns of data (filter length p, forgetting factor lambda)."""
    data = np.asarray(data, dtype=float)
    if data.ndim < 2:
        data = data.reshape(1, -1)
    nScans = data.shape[1]
    nPoints = data.shape[0]
    if direc is None:
        direc = 'fb'
    if delta is None:
        delta = 100
    if nPreAvg is None:
        nPreAvg = 0
    if lambda_ is None:
        raise ValueError('At least three inputs (data, p, lambda) are needed.')
    if nScans <= 1:
        raise ValueError('Multiple scans (columns) are needed. First input must be a 2D array with more than '
                         'one column.')
    if nPoints <= 1:
        raise ValueError('Multiple points (rows) are needed. First input must be a 2D array with individual '
                         'scans along columns.')
    if p >= nPoints:
        raise ValueError('Filter length (%d) is longer than data (%d), must be shorter.' % (p, nPoints))
    if lambda_ < 0 or lambda_ > 1:
        raise ValueError('lambda (third input) must be between 0 and 1. Typical values are around 0.96-0.98.')
    if p <= 0:
        raise ValueError('Filter length p (second input) must be positive. Typical values are 10-50.')
    if direc == 'f':
        fwd, bwd = True, False
    elif direc == 'b':
        fwd, bwd = False, True
    elif direc in ('fb', 'bf'):
        fwd, bwd = True, True
    else:
        raise ValueError("Unknown string '%s'in direction parameter" % direc)
    p = int(p)
    if nPreAvg == 0:
        d = np.mean(data, axis=1)
    else:
        nPreAvg = int(min(nPreAvg, nScans))
        d = np.mean(data[:, :nPreAvg], axis=1)
    nPreAvg = int(nPreAvg)
    if fwd:
        y_forward = _rls(data, d, p, lambda_, delta, nPreAvg)
    if bwd:
        y_backward = _rls(data[::-1, :], d[::-1], p, lambda_, delta, nPreAvg)[::-1]
    if fwd:
        y = y_forward
        if bwd:
            y = (y + y_backward) / 2
    else:
        y = y_backward
    y = y.reshape(-1, 1)
    if plot:
        import matplotlib.pyplot as plt
        tt = np.arange(1, nPoints + 1)
        plt.plot(tt, np.mean(data, axis=1), 'r', tt, y.ravel(), 'b')
        plt.title('System output')
        plt.legend(['conventional avg', 'filtered avg'])
        plt.show()
    return y
