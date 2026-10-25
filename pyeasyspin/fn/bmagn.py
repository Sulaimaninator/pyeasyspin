"""Bohr magneton, J/T (CODATA, as in EasySpin 6.0.12)."""
from .._registry import nout


@nout(2)
def bmagn():
    """(value, sigma): Bohr magneton, J/T and its standard uncertainty."""
    return 9.2740100657e-24, 0.0000000029e-24
