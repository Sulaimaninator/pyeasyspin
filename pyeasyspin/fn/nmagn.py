"""Nuclear magneton, J/T (CODATA, as in EasySpin 6.0.12)."""
from .._registry import nout


@nout(2)
def nmagn():
    """(value, sigma): Nuclear magneton, J/T and its standard uncertainty."""
    return 5.0507837393e-27, 0.0000000016e-27
