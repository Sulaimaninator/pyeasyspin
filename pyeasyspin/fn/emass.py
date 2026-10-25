"""Electron mass, kg (CODATA, as in EasySpin 6.0.12)."""
from .._registry import nout


@nout(2)
def emass():
    """(value, sigma): Electron mass, kg and its standard uncertainty."""
    return 9.1093837139e-31, 0.0000000028e-31
