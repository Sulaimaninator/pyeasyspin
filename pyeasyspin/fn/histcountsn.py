"""Histogram counts of multivariate data along each axis"""
import numpy as np
from .._registry import nout
from .._matlab import discretize as _discretize


@nout(2)
def histcountsn(X=None, edges=None):
    """histcountsn(X, edges) -> (n, bin): counts of the nPoints x nDims data X on the grid given by the
    list of edge vectors (n has size [numel(edges{d})-1 ...]); bin = 1-based bin subscripts of the
    in-range points (nIn x nDims)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    nPoints, nDims = X.shape[0], X.shape[1]
    inrange = np.ones(nPoints, dtype=bool)
    Xd, siz = [], []
    for d in range(nDims):
        Xd.append(_discretize(X[:, d], edges[d]))
        inrange[np.isnan(Xd[d])] = False
        siz.append(np.size(edges[d]) - 1)
    Xd = [x[inrange].astype(np.int64) for x in Xd]
    lin = np.zeros(int(inrange.sum()), dtype=np.int64)
    stride = 1
    for d in range(nDims):                      # sub2ind, column-major
        lin += (Xd[d] - 1) * stride
        stride *= siz[d]
    total = int(np.prod(siz))
    n = np.bincount(lin, minlength=total).astype(float)
    n = n.reshape(siz if len(siz) > 1 else (1, total), order='F')
    bin_ = np.stack(Xd, axis=1).astype(float) if Xd else np.zeros((0, 0))
    return n, bin_
