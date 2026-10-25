"""Bicubic spline interpolation of a matrix on its index grid (EasySpin's fast 2-D spline)."""
import numpy as np

from .._registry import F
from .._matlab import unmkpp


def esspline2d(data=None, ri=None, ci=None, EndCon=None):
    """Values of the bicubic spline through data (rows r = 1..nr, columns c = 1..nc) at the points
    (ri, ci). EndCon = [rows columns] end conditions as in esspline1d. The result has the shape of ri."""
    if EndCon is None:
        EndCon = [0, 0]
    EndCon = np.asarray(EndCon).ravel()
    ri_a = np.asarray(ri, dtype=float)
    points = np.vstack([ri_a.ravel(order='F'), np.asarray(ci, dtype=float).ravel(order='F')])
    n = ri_a.size
    _, coefs, rL, k, d = unmkpp(F.esspline1d(data, EndCon[1]))
    if k < 4:
        coefs = np.hstack([np.zeros((coefs.shape[0], 4 - k)), coefs])
    values = coefs.reshape((d, rL * 4), order='F').T
    sizeg = [0, rL * 4]
    _, coefs, cL, k, d = unmkpp(F.esspline1d(values, EndCon[0]))
    if k < 4:
        coefs = np.hstack([np.zeros((coefs.shape[0], 4 - k)), coefs])
    values = coefs.reshape((d, cL * 4), order='F').T
    sizeg[0] = cL * 4
    ipoint = np.fix(points)
    ipoint[0, ipoint[0, :] > cL] = cL
    ipoint[1, ipoint[1, :] > rL] = rL
    rpoint = points - ipoint
    ells = (sizeg[0] // 4, sizeg[1] // 4)
    i0 = ells[0] * np.arange(4)
    rows = np.concatenate([i0, i0, i0, i0]) + 1                    # 1-based subscripts
    cols = np.repeat(ells[1] * np.arange(4), 4) + 1
    offset = rows + sizeg[0] * (cols - 1)                          # sub2ind, 1-based
    base = ipoint[0, :] - 1 + sizeg[0] * (ipoint[1, :] - 1)
    lin = (base[None, :] + offset[:, None]).astype(int) - 1
    vflat = values.ravel(order='F')
    coefs = vflat[lin].reshape((4, 4, n), order='F')
    for i in (1, 0):                                               # MATLAB i = 2:-1:1
        s = rpoint[i, :].reshape(1, 1, n)
        coefs = coefs.reshape((4 ** i, 4, n), order='F')
        coefs = ((coefs[:, 0:1, :] * s + coefs[:, 1:2, :]) * s + coefs[:, 2:3, :]) * s + coefs[:, 3:4, :]
    return coefs.reshape(ri_a.shape if ri_a.ndim else (1,), order='F')
