"""Spin quantum numbers of a spin system."""
from .._registry import F


def spinvec(System=None):
    """Row vector of all spin quantum numbers (electrons, nuclei, orbital angular momenta)."""
    Sys, err = F.validatespinsys(System)
    if err:
        raise ValueError(err)
    return Sys.Spins
