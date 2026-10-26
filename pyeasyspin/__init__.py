"""pyeasyspin: a Python port of EasySpin v6.0.12 (MIT, Stefan Stoll and contributors).

Public EasySpin functions are attributes of this package (pyeasyspin.pepper, ...), loaded
lazily from pyeasyspin/fn/; `from pyeasyspin import *` imports all of them. Private helpers are
reachable through pyeasyspin._registry.F.

Public functions run with BLAS limited to config.BLASThreads threads (pyeasyspin/_parallel.py),
and return MATLAB row and column vectors as 1-D arrays (matrices keep MATLAB's shape;
config.MatlabShapes = True returns MATLAB's shapes unchanged)."""
import functools

import numpy as np

from ._struct import Struct
from .config import config
from ._public import PUBLIC
from ._parallel import blas_threads, parallel_map

__version__ = '0.1.0'
EASYSPIN_VERSION = '6.0.12'     # the EasySpin release this port follows
__all__ = ['Struct', 'config', 'parallel_map'] + list(PUBLIC)

_WRAPPED = {}


def _vec1d(x):
    """A 1 x N or N x 1 array -> 1-D; everything else unchanged."""
    if isinstance(x, np.ndarray) and x.ndim == 2 and 1 in x.shape and x.size > 1:
        return x.reshape(-1)
    return x


def _public(name):
    f = _WRAPPED.get(name)
    if f is None:
        from ._registry import F
        raw = F.get(name)

        @functools.wraps(raw)
        def f(*args, **kwargs):
            with blas_threads():
                out = raw(*args, **kwargs)
            if config.MatlabShapes:
                return out
            return tuple(_vec1d(o) for o in out) if isinstance(out, tuple) else _vec1d(out)
        f.__module__ = __name__          # pickled by reference as pyeasyspin.<name>
        f.__qualname__ = name
        _WRAPPED[name] = f
    return f


def __getattr__(name):
    if name in PUBLIC:
        return _public(name)
    raise AttributeError(f'module pyeasyspin has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(PUBLIC))
