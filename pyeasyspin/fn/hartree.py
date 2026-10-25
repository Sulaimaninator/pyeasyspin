"""Hartree energy, J (CODATA, as in EasySpin 6.0.12)."""
from .._registry import nout


@nout(2)
def hartree():
    """(value, sigma): Hartree energy, J and its standard uncertainty."""
    return 4.3597447222060e-18, 0.0000000000048e-18
