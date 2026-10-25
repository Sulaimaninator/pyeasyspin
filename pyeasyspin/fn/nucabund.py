"""Natural abundance(s), fraction."""
from .._registry import F


def nucabund(*varargin):
    """Natural abundance(s), fraction of the given isotopes (string or list)."""
    return F.nucdata(list(varargin))[3]
