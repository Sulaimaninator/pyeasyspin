"""pyeasyspin: a Python port of EasySpin v6.0.12 (MIT, Stefan Stoll and contributors).

Public EasySpin functions are attributes of this package (pyeasyspin.pepper, ...), loaded
lazily from pyeasyspin/fn/. Private helpers are reachable through pyeasyspin._registry.F."""
from ._struct import Struct
from .config import config
from ._public import PUBLIC

__version__ = '0.1.0'
EASYSPIN_VERSION = '6.0.12'     # the EasySpin release this port follows


def __getattr__(name):
    if name in PUBLIC:
        from ._registry import F
        return F.get(name)
    raise AttributeError(f'module pyeasyspin has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(PUBLIC))
