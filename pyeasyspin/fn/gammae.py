"""Electron gyromagnetic ratio, rad/s/T (CODATA, as in EasySpin 6.0.12)."""
from .._registry import nout


@nout(2)
def gammae():
    """(value, sigma): Electron gyromagnetic ratio, rad/s/T and its standard uncertainty."""
    return -1.76085962784e11, 0.00000000055e11
