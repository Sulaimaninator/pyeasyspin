"""Planck constant over 2 pi, J s."""
import math
from .._registry import F


def hbar():
    """Planck constant over 2 pi, J s."""
    return F.planck()/2/math.pi
