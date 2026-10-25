"""Faraday constant, C/mol."""
from .._registry import F


def faraday():
    """Faraday constant, C/mol."""
    return F.evolt()*F.avogadro()
