"""Simulation of/compensation for the effect of transmitter"""
import numpy as np
from .._struct import Struct, isfield
from .._mat import isempty, vec
from .._matlab import narrow as _narrow


def transmitter(signal0=None, Ain=None, Aout=None, option=None, *varargin):
    """transmitter(signal0, Ain, Aout, option) -> signal: transmitter nonlinearity (polynomial
    fit of Aout(Ain), then a not-a-knot spline), option 'simulate' or 'compensate'."""
    if signal0 is None:
        return None
    nargin = sum(a is not None for a in (signal0, Ain, Aout, option)) + len(varargin)
    if nargin != 4:
        raise ValueError('transmitter() requires 4 input arguments.')
    from scipy.interpolate import CubicSpline
    signal0 = np.asarray(signal0)
    shape0 = signal0.shape
    dim = shape0 if signal0.ndim >= 2 else (1, signal0.size)
    if min(dim) > 1:
        raise ValueError('Input must be a row or column vector.')
    if len(dim) > 2:
        raise ValueError('Input must be a row or column vector.')
    Ain = vec(np.asarray(Ain, dtype=float))
    Aout = vec(np.asarray(Aout, dtype=float))
    if Ain.size != Aout.size:
        raise ValueError('The input and output amplitude vectors need to have the same length.')
    if option == 'simulate' and np.max(np.abs(signal0)) > np.max(Ain):
        raise ValueError('The input signal amplitude needs to be defined on the provided input amplitude '
                         'scale of the transmitter.')
    if option == 'compensate' and np.max(np.abs(signal0)) > np.max(Aout):
        raise ValueError('The amplitude of the desired output signal needs to be within the provided output '
                         'amplitude scale of the transmitter.')
    Opt = Struct(varargin[0]) if nargin == 5 else Struct()
    if not isfield(Opt, 'N') or isempty(Opt.N):
        Opt.N = 4
    nN = int(Opt.N)

    # Vandermonde matrix without the constant column: V(:,j) = Ain.^(N-j), j = 1..N-1
    V = np.zeros((Ain.size, nN - 1))
    V[:, nN - 2] = Ain
    for j in range(nN - 3, -1, -1):
        V[:, j] = Ain * V[:, j + 1]
    coeff = np.concatenate((np.linalg.lstsq(V, Aout, rcond=None)[0], [0.0]))
    Aout = np.polyval(coeff, Ain)

    if option == 'simulate':
        Fi = CubicSpline(Ain, Aout, bc_type='not-a-knot')
    elif option == 'compensate':
        OutputAmplitude, uniqueind = np.unique(Aout, return_index=True)
        InputAmplitude = Ain[uniqueind]
        Fi = CubicSpline(OutputAmplitude, InputAmplitude, bc_type='not-a-knot')
    else:
        raise ValueError("Requested calculation option not available. Available options are 'simulate' "
                         "and 'compensate'.")

    s = vec(signal0)
    if not np.iscomplexobj(signal0):
        signal = np.sign(s) * Fi(np.abs(s))
    else:
        signal = np.sign(s.real) * Fi(np.abs(s.real)) + 1j * np.sign(s.imag) * Fi(np.abs(s.imag))
        signal = _narrow(signal)
    return signal.reshape(dim, order='F')
