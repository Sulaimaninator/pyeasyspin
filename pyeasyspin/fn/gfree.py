"""Free-electron g value (CODATA 2018)."""
from .._registry import nout


@nout(2)
def gfree():
    """(g, sigma): free-electron g value and its standard uncertainty."""
    g = 2.00231930436092
    sigma = 0.00000000000036
    return g, sigma
