"""Vacuum permittivity, F/m (CODATA, as in EasySpin 6.0.12)."""
from .._registry import nout


@nout(2)
def eps0():
    """(value, sigma): Vacuum permittivity, F/m and its standard uncertainty."""
    return 8.8541878188e-12, 0.0000000014e-12
