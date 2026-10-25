"""Convert magnetic field (mT) to frequency (MHz)."""
from .._registry import F


def mt2mhz(value_mT=None, g=None):
    """Frequency in MHz of a field value_mT (mT) for g (default: free-electron g)."""
    if g is None:
        g = F.gfree()[0]
    return F.unitconvert(value_mT, 'mT->MHz', g)
