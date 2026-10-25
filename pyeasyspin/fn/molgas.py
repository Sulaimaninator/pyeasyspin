"""Molar gas constant, J/(mol K)."""
from .._registry import F


def molgas():
    """Molar gas constant, J/(mol K)."""
    return F.avogadro()*F.boltzm()
