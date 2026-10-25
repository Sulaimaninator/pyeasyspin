"""Conversion from MHz to mT"""
from .._registry import F
from .._mat import isnumeric


def mhz2mt(value_MHz=None, g=None):
    """Field in mT corresponding to value_MHz (MHz) for g (default: free-electron g)."""
    if g is None:
        g = F.gfree()[0]
    if not isnumeric(g):
        raise ValueError('Second input (g) must be numeric.')
    return F.unitconvert(value_MHz, 'MHz->mT', g)
