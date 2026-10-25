"""Resonator profile"""
import numpy as np
from .._mat import isempty


def _H(nu, nu0, Q):
    """1./(1+1i*Q*(nu/nu0-nu0./nu)); an infinite argument (nu = 0) gives 0, as in MATLAB."""
    with np.errstate(divide='ignore', invalid='ignore'):
        x = Q * (nu / nu0 - nu0 / nu)
        out = 1 / (1 + 1j * x)
    out = np.asarray(out, dtype=complex)
    bad = np.isinf(x)
    if np.any(bad):
        out[bad] = complex(-0.0, 0.0)
    return out


def resonatorprofile(nu=None, nu0=None, Qu=None, beta=None, type=None, nargout=None, plot=False):
    """resonatorprofile(nu, nu0, Qu[, beta], type) -> out, or (nu, out) with nargout=2.
    type 'transferfunction', 'voltagereflection' or 'powerreflection'; nu, nu0 in GHz.
    Empty nu gives a 1001-point axis around nu0. plot=True draws the profile (MATLAB nargout=0)."""
    if nargout is None:
        nargout = 0 if plot else 1
    nargin = sum(a is not None for a in (nu, nu0, Qu, beta, type))
    if nargin == 0:
        return None
    if nargin == 4:
        if isinstance(beta, str):
            type = beta
            beta = np.zeros((0, 0))
        else:
            raise ValueError("Please specify the type of requested output type, 'transferfunction', "
                             "'voltagereflection', or 'powerreflection'.")
    if beta is None:
        beta = np.zeros((0, 0))
    if isempty(beta) and 'reflection' in type:
        raise ValueError('The coupling coefficient beta is required as a fourth input argument to '
                         'resonatorprofile().')
    nu0 = float(np.ravel(nu0)[0])
    Qu = float(np.ravel(Qu)[0])
    if isempty(nu):
        dnu = nu0 / Qu * 10
        from .._matlab import linspace as _linspace
        nu = _linspace(max(nu0 - dnu, 0), nu0 + dnu, 1001).reshape(1, -1)
    nu_in = nu
    nu = np.asarray(nu, dtype=float)

    if type == 'transferfunction':
        if not isempty(beta):
            Q = Qu / (1 + float(np.ravel(beta)[0]))
        else:
            Q = Qu
        out = _H(nu, nu0, Q)
    elif type == 'voltagereflection':
        b = float(np.ravel(beta)[0])
        Hq = _H(nu, nu0, Qu)
        out = (1 - b * Hq) / (1 + b * Hq)
    elif type == 'powerreflection':
        b = float(np.ravel(beta)[0])
        Hq = _H(nu, nu0, Qu)
        out = np.abs((1 - b * Hq) / (1 + b * Hq)) ** 2
    else:
        raise ValueError(f"Unknown type '{type}'.")

    if nargout == 0:
        import matplotlib.pyplot as plt
        x = np.ravel(nu)
        if type == 'transferfunction':
            plt.plot(x, np.real(np.ravel(out)))
            plt.ylabel('H, real part of the transfer function')
        elif type == 'voltagereflection':
            plt.plot(x, np.abs(np.ravel(out)))
            plt.ylabel('|Gamma|, norm of voltage reflection coefficient')
        else:
            plt.plot(x, np.ravel(out))
            plt.ylabel('P_reflected, power reflection coefficient')
        plt.xlabel('frequency (GHz)')
        plt.ylim(0, 1)
        plt.axvline(nu0, color='r')
        plt.show()
        return None
    if nargout == 1:
        return out
    if nargout == 2:
        return nu_in, out
    return None
