"""Rescaling of data"""
import numpy as np
from .._registry import nout
from .._mat import isempty, vec


@nout(2)
def rescaledata(*varargin):
    """rescaledata(y[, yref][, mode][, region]) -> (yscaled, scalefactor).
    Modes 'maxabs' (default without yref), 'lsq' (default with yref), 'int', 'dint', 'none'."""
    nargin = len(varargin)
    region = None
    yref = None
    if nargin == 1:
        y, Mode = varargin[0], 'maxabs'
    elif nargin == 2:
        y, input2 = varargin
        if isinstance(input2, str):
            Mode = input2
        else:
            yref, Mode = input2, 'lsq'
    elif nargin == 3:
        y, input2 = varargin[0], varargin[1]
        if isinstance(input2, str):
            Mode, region = input2, varargin[2]
        else:
            yref, Mode = input2, varargin[2]
    elif nargin == 4:
        y, yref, Mode, region = varargin
    else:
        raise ValueError('Wrong number of input parameters.')
    y = np.asarray(y)
    if y.ndim < 2:
        y = y.reshape(1, -1)
    if y.ndim > 2 or min(y.shape) != 1:
        raise ValueError('y must be a row or column vector. rescaledata() does not work on 2D arrays.')
    isRowVector = y.shape[0] == 1
    if isempty(yref):
        yref = None
    if yref is not None:
        if isinstance(yref, str):
            raise ValueError('Second input must be the reference vector.')
        ya = np.asarray(yref)
        if ya.ndim > 2 or (ya.ndim == 2 and min(ya.shape) != 1):
            raise ValueError('yref must be a row or column vector.')
    Modes = ['maxabs', 'lsq', 'int', 'dint', 'none']
    if Mode not in Modes:
        if Mode in ('lsq0', 'lsq1', 'lsq2', 'minmax'):
            raise ValueError("Mode '%s' is obsolete. Use basecorr() instead." % Mode)
        raise ValueError("Unknown scaling mode '%s'." % Mode)
    ModeID = Modes.index(Mode) + 1
    if ModeID == 2:
        if yref is None:
            raise ValueError("For '%s' mode, a reference is needed." % Mode)
        if y.size != np.size(yref):
            from scipy.interpolate import Akima1DInterpolator
            yr = vec(np.asarray(yref, dtype=float))
            yref = Akima1DInterpolator(np.linspace(0, 1, yr.size), yr, method='makima')(np.linspace(0, 1, y.size))
    y = vec(y)
    if yref is not None:
        yref = vec(np.asarray(yref))
    reg = None if isempty(region) else vec(np.asarray(region)).astype(bool)
    if ModeID == 1:
        idx = ~np.isnan(y)
        if reg is not None:
            idx = idx & reg
        scalefactor = 1 / np.max(np.abs(y[idx]))
        if yref is not None:
            idx = ~np.isnan(yref)
            if reg is not None:
                idx = idx & reg
            scalefactor = np.max(np.abs(yref[idx])) * scalefactor
    elif ModeID == 2:
        idx = ~np.isnan(y) & ~np.isnan(yref)
        if reg is not None:
            idx = idx & reg
        a, b = yref[idx], y[idx]
        scalefactor = 1 / (np.vdot(a, b) / np.vdot(a, a))    # yref\y (least squares, scalar)
    elif ModeID == 3:
        idx = ~np.isnan(y)
        if reg is not None:
            idx = idx & reg
        scalefactor = 1 / np.sum(y[idx])
    elif ModeID == 4:
        idx = ~np.isnan(y)
        if reg is not None:
            idx = idx & reg
        scalefactor = 1 / np.sum(np.cumsum(y[idx]))
    else:
        scalefactor = 1
    if np.real(scalefactor) < 0:
        scalefactor = abs(scalefactor)
    yscaled = y * scalefactor
    yscaled = yscaled.reshape(1, -1) if isRowVector else yscaled.reshape(-1, 1)
    return yscaled, scalefactor
