"""Nuclear g value(s)."""
from .._registry import F


def nucgval(*varargin):
    """Nuclear g value(s) of the given isotopes (string or list)."""
    return F.nucdata(list(varargin))[1]
