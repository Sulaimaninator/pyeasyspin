"""Gaussian and Lorentzian line shapes"""
import numpy as np
from .._registry import F


def lshape(x=None, x0=None, fwhm=None, *varargin):
    """lshape(x, x0, fwhm, diff=0, alphaGauss=1, phase=0) -> y (pseudo-Voigt mix)."""
    diff, alphaGauss, phase = 0, 1, 0
    if len(varargin) >= 1:
        diff = varargin[0]
    if len(varargin) >= 2:
        alphaGauss = varargin[1]
    if len(varargin) >= 3:
        phase = varargin[2]
    fwhm = np.ravel(np.asarray(fwhm))
    if np.any(fwhm <= 0) or np.iscomplexobj(fwhm):
        raise ValueError('fwhm must be positive and real!')
    if fwhm.size > 2:
        raise ValueError('fwhm must contain 1 or 2 elements!')
    fwhmG = float(fwhm[0])
    fwhmL = float(fwhm[1]) if fwhm.size == 2 else float(fwhm[0])
    if alphaGauss != 0:
        GaussAbs, GaussDisp = F.gaussian(x, x0, fwhmG, diff, nargout=2)
        Gauss = GaussAbs * np.cos(phase) + GaussDisp * np.sin(phase) if phase != 0 else GaussAbs
    else:
        Gauss = 0
    if alphaGauss != 1:
        LorentzAbs, LorentzDisp = F.lorentzian(x, x0, fwhmL, diff, nargout=2)
        Lorentz = LorentzAbs * np.cos(phase) + LorentzDisp * np.sin(phase) if phase != 0 else LorentzAbs
    else:
        Lorentz = 0
    return alphaGauss * Gauss + (1 - alphaGauss) * Lorentz
