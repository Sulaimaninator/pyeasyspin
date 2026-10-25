"""Neutron mass, kg (CODATA, as in EasySpin 6.0.12)."""
from .._registry import nout


@nout(2)
def nmass():
    """(value, sigma): Neutron mass, kg and its standard uncertainty."""
    return 1.67492750056e-27, 0.00000000085e-27
