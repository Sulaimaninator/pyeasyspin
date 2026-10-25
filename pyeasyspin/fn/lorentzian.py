"""Lorentzian line shape"""
import numpy as np
from .._registry import nout


@nout(-1)
def lorentzian(x=None, x0=None, fwhm=None, diff=None, phase=None, nargout=None):
    """lorentzian(x, x0, fwhm, diff=0, phase=0) -> y; nargout=2 -> (y1, y2) (absorption, dispersion)."""
    if x is None or x0 is None or fwhm is None:
        raise ValueError('lorentzian() needs at least three inputs: x, x0, and fwhm.')
    if diff is None:
        diff = 0
    if phase is None:
        phase = 0
    if np.size(x0) != 1 or np.iscomplexobj(x0):
        raise ValueError('2nd input (x0, center) must be a real number.')
    if np.size(fwhm) != 1 or np.iscomplexobj(fwhm) or float(np.ravel(fwhm)[0]) <= 0:
        raise ValueError('3rd input (fwhm, full width at half maximum) must be a real and positive number.')
    if np.size(diff) != 1 or diff < -1 or diff > 2 or np.mod(diff, 1):
        raise ValueError('4th input (diff, lineshape derivative) must be -1, 0, 1, or 2.')
    if np.size(phase) != 1 or np.iscomplexobj(phase):
        raise ValueError('5th input (phase) must be a real number.')
    x0, fwhm, phase = float(np.ravel(x0)[0]), float(np.ravel(fwhm)[0]), float(np.ravel(phase)[0])
    diff = int(np.ravel(diff)[0])
    x = np.asarray(x, dtype=float)
    gamma = fwhm / np.sqrt(3)
    pre = 2 / np.pi / np.sqrt(3)
    z = (x - x0) / gamma
    if diff == -1:
        yabs = np.arctan(2 / np.sqrt(3) * z) / np.pi + 1 / 2
        ydisp = (1 / 2 / np.pi) * np.log(3 + 4 * z ** 2)
    elif diff == 0:
        yabs = pre / gamma / (1 + 4 / 3 * z ** 2)
        ydisp = pre ** 2 * np.pi / gamma * z / (1 + 4 / 3 * z ** 2)
    elif diff == 1:
        yabs = -8 / 3 * pre / gamma ** 2 * z / (1 + 4 / 3 * z ** 2) ** 2
        ydisp = pre ** 2 * np.pi / gamma ** 2 * (1 - 4 / 3 * z ** 2) / (1 + 4 / 3 * z ** 2) ** 2
    else:
        yabs = 8 / 3 * pre / gamma ** 3 * (4 * z ** 2 - 1) / (1 + 4 / 3 * z ** 2) ** 3
        ydisp = pre ** 2 * np.pi / gamma ** 3 * 2 * 4 / 3 * z * (4 / 3 * z ** 2 - 3) / (1 + 4 / 3 * z ** 2) ** 3
    if phase != 0:
        y1 = yabs * np.cos(phase) + ydisp * np.sin(phase)
        y2 = -yabs * np.sin(phase) + ydisp * np.cos(phase)
    else:
        y1, y2 = yabs, ydisp
    if nargout is not None and nargout > 1:
        return y1, y2
    return y1
