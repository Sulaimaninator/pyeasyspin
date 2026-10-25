"""kron(A, eye(n))."""
import numpy as np
import scipy.sparse as sps


def kroneye(*varargin):
    """kroneye(A) or kroneye(A, nI[, manual]): kron(A, eye(nI)); sparse in, sparse out."""
    A = varargin[0]
    if len(varargin) == 1:
        nI = np.shape(A)[0]
        if np.shape(A)[1] != nI:
            raise ValueError('Input matrix must be square.')
    else:
        nI = varargin[1]
    if np.size(nI) != 1:
        raise ValueError('Second input (nI) must be square')
    nI = int(nI)
    if sps.issparse(A):
        return sps.kron(A, sps.identity(nI, format='csc'), format='csc')
    return np.kron(np.atleast_2d(A), np.eye(nI))
