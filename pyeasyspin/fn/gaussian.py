"""Gaussian line shape"""
import numpy as np
from .._registry import F, nout


def _isscalar(v):
    return np.size(v) == 1


def _isreal(v):
    return not np.iscomplexobj(v)


@nout(-1)
def gaussian(x=None, x0=None, fwhm=None, diff=None, phase=None, nargout=None):
    """gaussian(x, x0, fwhm, diff=0, phase=0) -> y (absorption, area-normalized);
    nargout=2 -> (y1, y2) with y2 the dispersion (quadrature) signal."""
    if x is None or x0 is None or fwhm is None:
        raise ValueError('gaussian() needs at least three inputs: x, x0, and fwhm.')
    if diff is None:
        diff = 0
    if phase is None:
        phase = 0
    if not _isscalar(x0) or not _isreal(x0):
        raise ValueError('2nd input (x0, center) must be a real number.')
    if not _isscalar(fwhm) or not _isreal(fwhm) or float(np.real(fwhm)) <= 0:
        raise ValueError('3rd input (fwhm, full width at half maximum) must be a real and positive number.')
    if not _isscalar(diff) or diff < -1 or np.mod(diff, 1):
        raise ValueError('4th input (diff, lineshape derivative) must be -1, 0, 1, 2, etc.')
    if not _isscalar(phase) or not _isreal(phase):
        raise ValueError('5th input (phase) must be a real number.')
    x0, fwhm, phase = float(np.ravel(x0)[0]), float(np.ravel(fwhm)[0]), float(np.ravel(phase)[0])
    diff = int(np.ravel(diff)[0])
    x = np.asarray(x, dtype=float) if not np.iscomplexobj(x) else np.asarray(x)
    nonzeroPhase = np.mod(phase, 2 * np.pi) != 0
    returnDispersion = nargout is not None and nargout > 1
    calcDispersion = nonzeroPhase or returnDispersion

    sig = fwhm / np.sqrt(2 * np.log(2)) / 2
    k = (x - x0) / sig / np.sqrt(2)
    ydisp = None
    if diff == -1:
        from scipy.special import erf
        yabs = 1 / 2 * (1 + erf(k))
        if calcDispersion:
            # MATLAB uses hypergeom (Symbolic Math Toolbox): k^2*2F2(1,1;3/2,2;-k^2) = 2*int_0^k dawsn
            ydisp = np.sqrt(2 / np.pi) * 2 / np.sqrt(np.pi) / sig / 2 * sig * np.sqrt(2) * _dawson_int(k)
    else:
        yabs = (np.sqrt(2 / np.pi) / 2 / sig * (-1 / sig) ** diff * 2 ** (-diff / 2)
                * _hermitepoly(k, diff) * np.exp(-k ** 2))
        if calcDispersion:
            ydisp = (np.sqrt(2 / np.pi) * 2 / np.sqrt(np.pi) / sig / 2 * (1 / np.sqrt(2) / sig) ** diff
                     * _dawsonFderiv(k, diff))
    yabs = np.broadcast_to(yabs, np.shape(x)).copy() if np.ndim(yabs) < np.ndim(x) else yabs
    if nonzeroPhase:
        y1 = yabs * np.cos(phase) + ydisp * np.sin(phase)
        y2 = -yabs * np.sin(phase) + ydisp * np.cos(phase)
    else:
        y1 = yabs
        y2 = ydisp
    if returnDispersion:
        return y1, y2
    return y1


def _dawson_int(k):
    """1/2*k^2*2F2(1,1;3/2,2;-k^2) = int_0^k dawsn(t) dt (MATLAB: hypergeom)."""
    from scipy.special import dawsn
    from scipy.integrate import quad
    k = np.asarray(k, dtype=float)
    out = np.array([quad(dawsn, 0.0, float(v))[0] for v in k.ravel()])
    return out.reshape(k.shape)


def _dawsonF(x):
    y = np.sqrt(np.pi) / 2j * (F.faddeeva(x) - np.exp(-x ** 2))
    return np.real(y)


def _dawsonFderiv(x, n):
    return (-1) ** n * (_hermitepoly(x, n) * _dawsonF(x) - _Gpoly(x, n - 1))


def _Gpoly(x, n):
    if n == -1:
        return 0
    if n == 0:
        return 1
    if n == 1:
        return 2 * x
    return 2 * x * _Gpoly(x, n - 1) - 2 * n * _Gpoly(x, n - 2)


def _hermitepoly(x, n):
    if n == 0:
        return 1
    if n == 1:
        return 2 * x
    return 2 * x * _hermitepoly(x, n - 1) - 2 * (n - 1) * _hermitepoly(x, n - 2)
