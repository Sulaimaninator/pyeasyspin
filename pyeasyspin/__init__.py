"""pyeasyspin: a Python port of EasySpin v6.0.12 (MIT, Stefan Stoll and contributors).

Public EasySpin functions are attributes of this package (pyeasyspin.pepper, ...), loaded
lazily from pyeasyspin/fn/. Private helpers are reachable through pyeasyspin._registry.F.
Public functions run with BLAS limited to config.BLASThreads threads, and independent
simulations use all CPU cores (config.Workers; see pyeasyspin/_parallel.py)."""
import functools

from ._struct import Struct
from .config import config
from ._public import PUBLIC
from ._parallel import blas_threads, parallel_map

__version__ = '0.1.0'
EASYSPIN_VERSION = '6.0.12'     # the EasySpin release this port follows

_WRAPPED = {}


def _public(name):
    f = _WRAPPED.get(name)
    if f is None:
        from ._registry import F
        raw = F.get(name)

        @functools.wraps(raw)
        def f(*args, **kwargs):
            with blas_threads():
                return raw(*args, **kwargs)
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
