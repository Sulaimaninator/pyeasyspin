"""Proton mass, kg (CODATA, as in EasySpin 6.0.12)."""
from .._registry import nout


@nout(2)
def pmass():
    """(value, sigma): Proton mass, kg and its standard uncertainty."""
    return 1.67262192595e-27, 0.00000000052e-27
