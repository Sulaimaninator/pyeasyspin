"""Rydberg constant, 1/m (CODATA, as in EasySpin 6.0.12)."""
from .._registry import nout


@nout(2)
def rydberg():
    """(value, sigma): Rydberg constant, 1/m and its standard uncertainty."""
    return 10973731.568157, 0.000012
