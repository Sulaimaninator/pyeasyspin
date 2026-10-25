"""Atomic mass unit, kg (CODATA, as in EasySpin 6.0.12)."""
from .._registry import nout


@nout(2)
def amu():
    """(value, sigma): Atomic mass unit, kg and its standard uncertainty."""
    return 1.66053906892e-27, 0.00000000052e-27
