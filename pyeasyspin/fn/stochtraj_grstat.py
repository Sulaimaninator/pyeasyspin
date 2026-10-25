"""Calculate the Gelman-Rubin R statistic for a set of Markov chains"""
import numpy as np


def _find_gr(X):
    n, m = X.shape
    B = n * np.var(np.mean(X, axis=0), ddof=1) if m > 1 else 0.0
    W = np.mean(np.var(X, axis=0, ddof=1)) if n > 1 else 0.0
    sigmasq = (n - 1) / n * W + B / n
    Vhat = sigmasq + B / m / n
    return Vhat / W


def stochtraj_grstat(A=None):
    """stochtraj_grstat(A) -> stat: Gelman-Rubin R of a 2-D array (scalar) or of each component
    A(k,:,:) of a 3-D array (column, one value per component)."""
    if A is None:
        return None
    if not isinstance(A, np.ndarray) and not isinstance(A, (int, float, list)):
        raise ValueError('Must provide a numeric array for input.')
    A = np.asarray(A)
    if not np.issubdtype(A.dtype, np.number):
        raise ValueError('Must provide a numeric array for input.')
    if A.ndim < 2:
        A = A.reshape(1, -1)
    while A.ndim > 3 and A.shape[-1] == 1:
        A = A.reshape(A.shape[:-1])
    ndimsA = A.ndim
    if ndimsA > 3:
        raise ValueError('Array dimensions greater than 3 are not supported.')
    # EasySpin writes ~sizeA(2)>1, i.e. (~sizeA(2))>1, which is never true: only isrow/iscolumn act.
    if ndimsA == 2 and A.shape[0] == 1:
        raise ValueError('Markov chain length must be greater than one.')
    if ndimsA == 2 and A.shape[1] == 1:
        raise ValueError('Must have more than one Markov chain.')
    A = A.astype(float)
    if ndimsA == 2:
        return float(_find_gr(A))
    nComps = A.shape[0]
    stat = np.zeros((nComps, 1))
    for iComp in range(nComps):
        X = A[iComp]
        if X.shape[0] == 1:            # squeeze(1x1xm) is m x 1
            X = X.reshape(-1, 1)
        stat[iComp, 0] = _find_gr(X)
    return stat
