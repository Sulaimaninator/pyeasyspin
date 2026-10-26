"""Cubic spline interpolation of rows of data on the grid 1..n (EasySpin's fast spline)."""
import numpy as np

from .._mat import vec
from .._matlab import linspace, mkpp, fix


def esspline1d(y=None, EndCon=None, xx=None):
    """Rows of y (nDims x n) are splined over 1..n. EndCon 0: not-a-knot-like (EasySpin's
    variant), 1: zero end slopes, 2: periodic. Without xx: the pp struct (as mkpp). With xx: values
    at the points xx, or at linspace(1, n, xx) if xx is a scalar (nDims x nn)."""
    if EndCon is None:
        EndCon = 0
    EndCon = int(np.asarray(EndCon).ravel()[0])
    y = np.atleast_2d(np.asarray(y)).T.copy()          # y = y.'
    n, nDims = y.shape
    if n < 2:
        raise ValueError('Interpolation not possible: n < 2.')
    dy = np.diff(y, axis=0)
    if n == 2:
        if EndCon != 1:
            x = np.array([1.0, 2.0])
            c = np.column_stack([np.zeros((nDims, 2)), dy.T, y[0, :]])
        else:
            dy2 = np.diff(np.vstack([np.zeros((1, nDims)), dy, np.zeros((1, nDims))]), axis=0)
            x = np.arange(1.0, n + 1)
            c = np.column_stack([np.diff(dy2, axis=0).T, (np.array([[2.0, -1.0]]) @ dy2).T,
                                 np.zeros((nDims, 1)), y[0, :]])
    elif n == 3 and EndCon == 0:
        y[1:3, :] = dy
        y[2, :] = np.diff(dy, axis=0) / 2
        y[1, :] = y[1, :] - y[2, :]
        x = np.array([1.0, 3.0])
        c = np.column_stack([np.zeros((nDims, 1)), y[[2, 1, 0], :].T])
    else:
        A = np.zeros((n, n))
        i = np.arange(1, n - 1)
        A[i, i - 1] = 1.0
        A[i, i + 1] = 1.0
        A[i, i] = 4.0
        A[0, 0] = 1.0
        A[n - 1, n - 1] = 1.0
        b = np.zeros((n, nDims), dtype=np.result_type(y.dtype, float))
        b[1:n - 1, :] = 3 * (dy[0:n - 2, :] + dy[1:n - 1, :])
        if EndCon == 0:
            A[0, 1] = 2.0
            A[n - 1, n - 2] = 2.0
            b[0, :] = 0.5 * (5 * dy[0, :] + dy[1, :])
            b[n - 1, :] = 0.5 * (dy[n - 3, :] + 5 * dy[n - 2, :])
        elif EndCon == 1:
            A[0, 1] = 0.0
            A[n - 1, n - 2] = 0.0
        elif EndCon == 2:
            A[0, n - 1] = -1.0
            A[n - 1, [0, 1, n - 2, n - 1]] = [2.0, 1.0, 1.0, 2.0]
            b[n - 1, :] = 3 * (dy[0, :] + dy[n - 2, :])
        slp = np.linalg.solve(A, b)
        c4 = slp[0:n - 1, :] + slp[1:n, :] - 2 * dy[0:n - 1, :]
        c3 = dy[0:n - 1, :] - slp[0:n - 1, :] - c4
        x = np.arange(1.0, n + 1)
        M = np.hstack([c4.T, c3.T, slp[0:n - 1, :].T, y[0:n - 1, :].T])    # nDims x 4(n-1)
        c = M.reshape(((n - 1) * nDims, 4), order='F')
    if xx is None:
        return mkpp(x, c, nDims)
    xxa = np.asarray(xx, dtype=float)
    if xxa.size <= 1:
        nn = int(xxa.ravel()[0]) if xxa.size else 0
        xs = linspace(1, n, nn)
    else:
        nn = xxa.size
        xs = xxa.ravel(order='F')
    npp = x.size - 1
    index = fix(xs)
    index[index > npp] = npp
    xs = xs - index
    if nDims > 1:
        xs = np.tile(xs, (nDims, 1)).reshape(-1, order='F')
        index = nDims * index
        temp = np.arange(-nDims, 0).reshape(-1, 1)
        index = (1 + np.tile(index, (nDims, 1)) + temp).reshape(-1, order='F')
    cc = c[index.astype(int) - 1, :]
    yy = cc[:, 0]
    for i in range(1, 4):
        yy = xs * yy + cc[:, i]
    return yy.reshape((nDims, nn), order='F')
