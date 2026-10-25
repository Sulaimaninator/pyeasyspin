"""Nuclear spin quantum number(s)."""
from .._registry import F


def nucspin(*varargin):
    """Nuclear spin quantum number(s) of the given isotopes (string or list)."""
    return F.nucdata(list(varargin))[0]
