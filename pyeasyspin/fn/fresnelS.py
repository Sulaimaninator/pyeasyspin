"""- Fresnel sine integrals, S(X), S1(X), or S2(X)"""
from .fresnelC import _fresnel_core


def fresnelS(X=None, fresnelType=None):
    """fresnelS(X, fresnelType=0): S(X) = int_0^X sin(pi t^2/2) dt (type 0); type 1 and 2 are the
    S1/S2 variants. EasySpin's spline table on [0, 7.5], asymptotic series above."""
    if X is None:
        raise ValueError('1 or 2 arguemtns are required.')
    return _fresnel_core(X, fresnelType, 'S', 'FSspl')
