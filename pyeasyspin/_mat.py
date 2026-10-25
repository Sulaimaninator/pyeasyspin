"""MATLAB-semantics helpers shared by all ported functions.

Column-major discipline: every (:), reshape and kron boundary goes through these helpers."""
import numpy as np


def isempty(x):
    """MATLAB isempty: None, '', [], 0-size arrays, empty dicts/lists are empty."""
    if x is None:
        return True
    if isinstance(x, (str, list, tuple, dict)):
        return len(x) == 0
    if isinstance(x, np.ndarray):
        return x.size == 0
    return False


def numel(x):
    if x is None:
        return 0
    if isinstance(x, (str, list, tuple)):
        return len(x)
    return np.size(x)


def vec(x):
    """MATLAB x(:) as a 1-D array (column-major)."""
    return np.asarray(x).ravel(order='F')


def col(x):
    """MATLAB x(:) as an (n,1) column (column-major)."""
    return np.asarray(x).reshape(-1, 1, order='F')


def row(x):
    """MATLAB x(:).' as a (1,n) row (column-major)."""
    return np.asarray(x).reshape(1, -1, order='F')


def reshape(x, *shape):
    """MATLAB reshape (column-major). A single [] / -1 entry is inferred."""
    if len(shape) == 1 and not np.isscalar(shape[0]):
        shape = tuple(shape[0])
    shape = tuple(-1 if (s is None or (isinstance(s, (list, np.ndarray)) and np.size(s) == 0)) else int(s)
                  for s in shape)
    return np.reshape(np.asarray(x), shape, order='F')


def size(x, dim=None):
    """MATLAB size: at least two dims; dim is 1-based."""
    s = np.shape(x)
    if isinstance(x, str):
        s = (1, len(x)) if x else (0, 0)
    elif len(s) == 0:
        s = (1, 1)
    elif len(s) == 1:
        s = (1, s[0])
    if dim is None:
        return s
    return s[dim - 1] if dim <= len(s) else 1


def kron(a, b):
    """MATLAB kron (same convention as numpy.kron for 2-D operands)."""
    return np.kron(a, b)


def a2(x, dtype=None):
    """MATLAB value as a 2-D ndarray in MATLAB shape: None -> (0,0), scalar -> (1,1),
    1-D list/array -> (1,n) row, N-D kept. Sparse matrices are densified."""
    if x is None:
        return np.zeros((0, 0))
    if hasattr(x, 'toarray'):
        x = x.toarray()
    a = np.asarray(x) if dtype is None else np.asarray(x, dtype=dtype)
    if a.dtype == bool:
        a = a.astype(float)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(1, -1) if a.size else np.zeros((0, 0), dtype=a.dtype)
    return a


def issize(x, siz):
    """MATLAB isequal(size(x), siz) for 2-D sizes."""
    return tuple(size(x)) == tuple(int(s) for s in siz)


def isnumeric(x):
    """MATLAB isnumeric (bool counts as numeric here only for arrays of numbers)."""
    if isinstance(x, (bool, np.bool_)):
        return False
    if isinstance(x, (int, float, complex, np.number)):
        return True
    if hasattr(x, 'toarray'):
        return True
    if isinstance(x, np.ndarray):
        return x.dtype.kind in 'iufc'
    if isinstance(x, (list, tuple)):
        try:
            return np.asarray(x).dtype.kind in 'iufc'
        except Exception:
            return False
    return False
