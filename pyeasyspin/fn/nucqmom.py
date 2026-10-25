"""Nuclear electric quadrupole moment(s), barn."""
from .._registry import F


def nucqmom(*varargin):
    """Nuclear electric quadrupole moment(s), barn of the given isotopes (string or list)."""
    return F.nucdata(list(varargin))[2]
