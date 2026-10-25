"""Vacuum permeability, N/A^2 (CODATA, as in EasySpin 6.0.12)."""
from .._registry import nout


@nout(2)
def mu0():
    """(value, sigma): Vacuum permeability, N/A^2 and its standard uncertainty."""
    return 1.25663706127e-6, 0.00000000020e-6
