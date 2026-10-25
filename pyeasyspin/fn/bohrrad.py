"""Bohr radius, m (CODATA, as in EasySpin 6.0.12)."""
from .._registry import nout


@nout(2)
def bohrrad():
    """(value, sigma): Bohr radius, m and its standard uncertainty."""
    return 5.29177210544e-11, 0.00000000082e-11
