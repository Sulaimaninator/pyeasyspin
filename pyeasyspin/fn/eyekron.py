"""kron(eye(n), A)."""
import numpy as np
import scipy.sparse as sps


def eyekron(*varargin):
    """eyekron(A) or eyekron(nI, A[, manual]): kron(eye(nI), A); sparse in, sparse out."""
    if len(varargin) == 1:
        A = varargin[0]
        nI = np.shape(A)[0]
        if np.shape(A)[1] != nI:
            raise ValueError('Input matrix must be square.')
    else:
        nI, A = varargin[0], varargin[1]
    if np.size(nI) != 1:
        raise ValueError('First input (nI) must be square')
    nI = int(nI)
    if sps.issparse(A):
        return sps.kron(sps.identity(nI, format='csc'), A, format='csc')
    return np.kron(np.eye(nI), np.atleast_2d(A))
