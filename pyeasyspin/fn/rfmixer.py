"""Digital up- or downconversion"""
import math
import warnings
import numpy as np
from .._registry import F, nout
from .._struct import Struct, isfield
from .._mat import vec
from .._matlab import colon as _colon, interp1 as _interp1, cmax_index as _cmax_index


@nout(2)
def rfmixer(*varargin, nargout=None, plot=False):
    """rfmixer(t, signal, mwFreq, type[, Opt]) -> (tOut, signalOut). t in µs, mwFreq (LO) in GHz;
    type 'DSB', 'USB', 'LSB', 'IQmod', 'IQdemod' or 'IQshift'. Outputs are 1 x N rows.
    plot=True draws input and output (MATLAB nargout=0)."""
    if nargout is None:
        nargout = 0 if plot else 2
    nargin = len(varargin)
    if nargin == 0:
        return None
    if nargin in (1, 2, 3):
        raise ValueError('rfmixer() requires at least four input arguments.')
    elif nargin == 4:
        t, signal, mwFreq, type_ = varargin
        Opt = Struct()
    elif nargin == 5:
        t, signal, mwFreq, type_, Opt = varargin
        Opt = Struct(Opt)
    else:
        raise ValueError('rfmixer() cannot take %d input arguments.' % nargin)
    mwFreq = float(np.ravel(mwFreq)[0])
    signal = np.asarray(signal)
    if signal.ndim < 2:
        signal = signal.reshape(1, -1)
    t = np.asarray(t, dtype=float)
    if t.ndim < 2:
        t = t.reshape(1, -1)

    tl = type_.lower()
    if tl in ('dsb', 'usb', 'lsb', 'iqdemod'):
        if np.iscomplexobj(signal):
            raise ValueError('A real input signal is required for the selected mixer type.')
    elif tl in ('iqmod', 'iqshift'):
        if not np.iscomplexobj(signal):
            signal = signal + 1j * np.ones(signal.shape) * 1e-300
    else:
        raise ValueError('The selected mixer type is not available.')

    realInput = not np.iscomplexobj(signal)
    dim = signal.shape
    if min(dim) > 1 or len(dim) > 2:
        raise ValueError('Input must be a row or column vector.')
    if t.size != signal.size:
        raise ValueError('The signal and its time axis must have equal lengths.')
    if t.shape != signal.shape:
        t = t.reshape(signal.shape, order='F')

    if not isfield(Opt, 'BandwidthThreshold'):
        Opt.BandwidthThreshold = 0.1
    if not isfield(Opt, 'InterpolationMethod'):
        Opt.InterpolationMethod = 'spline'
    if not isfield(Opt, 'OverSampleFactor'):
        Opt.OverSampleFactor = 1.25
    if not isfield(Opt, 'NoiseCutoffThreshold'):
        Opt.NoiseCutoffThreshold = 0.1
    if not isfield(Opt, 'HilbertThreshold'):
        Opt.HilbertThreshold = 0.05

    f = vec(F.fdaxis(t))
    signalFT = np.fft.fftshift(np.fft.fft(vec(signal)))
    aFT = np.abs(signalFT)
    inputband = f[aFT > Opt.BandwidthThreshold * np.max(aFT)]
    maxFreqIn = np.max(inputband)

    if abs(mwFreq * 1e3) > 2 * maxFreqIn:
        maxFreqOut = abs(mwFreq * 1e3) + maxFreqIn
    else:
        maxFreqOut = max(abs(mwFreq * 1e3), maxFreqIn)
    nyqdt = 1 / (2 * maxFreqOut)

    tIn = vec(t)
    dtIn = tIn[1] - tIn[0]
    if isfield(Opt, 'dt'):
        if Opt.dt > nyqdt:
            warnings.warn('Maximum frequency exceeds Nyquist frequency for specified resampling time step.')
    else:
        if dtIn > nyqdt:
            Opt.dt = 1 / (2 * Opt.OverSampleFactor * maxFreqOut)
        else:
            Opt.dt = dtIn

    sig = vec(signal)
    if realInput:
        if tl in ('usb', 'lsb', 'iqdemod'):
            sig = sig + 1j * np.imag(vec(F.hilberttrans(signal)))
            asig = np.abs(sig)
            ind = np.flatnonzero(asig > Opt.NoiseCutoffThreshold * np.max(asig))
            signal_nonzero = sig[ind[0]:ind[-1] + 1]
            amplitude_FT = np.fft.fft(np.abs(signal_nonzero))
            cosphase_FT = np.fft.fft(np.cos(np.angle(signal_nonzero)))
            amplitude_FT = amplitude_FT / amplitude_FT[_cmax_index(amplitude_FT)]
            cosphase_FT = cosphase_FT / cosphase_FT[_cmax_index(cosphase_FT)]
            if np.max(np.abs(amplitude_FT * cosphase_FT)) > Opt.HilbertThreshold:
                raise ValueError('SSB conversion and digital downconversion require the quadrature signal'
                                 ' of the real input signal, which is calculated internally using the Hilbert'
                                 ' transform. In this case, it cannot be recovered. (Use the quadrature signal'
                                 ' as input or increase Opt.HilbertThreshold to still perform the calculation.)')

    if tl == 'usb':
        LO = abs(mwFreq)
    elif tl in ('lsb', 'iqdemod'):
        LO = -abs(mwFreq)
    else:
        LO = mwFreq

    signalOut_ = sig * np.exp(2j * math.pi * LO * 1e3 * tIn)
    tOut = _colon(tIn[0], Opt.dt, tIn[-1])
    signalOut = _interp1(tIn, signalOut_, tOut, Opt.InterpolationMethod)
    if tl in ('dsb', 'usb', 'lsb', 'iqmod'):
        signalOut = np.real(signalOut)
    tOut = tOut.reshape(1, -1)
    signalOut = signalOut.reshape(1, -1)

    if nargout == 0:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(tIn, np.real(sig))
        if not realInput:
            ax[0].plot(tIn, np.imag(sig))
        ax[0].set_title('Input signal')
        ax[1].plot(tOut.ravel(), np.real(signalOut).ravel())
        if np.iscomplexobj(signalOut):
            ax[1].plot(tOut.ravel(), np.imag(signalOut).ravel())
        ax[1].set_title('Output signal')
        ax[1].set_xlabel('t (µs)')
        plt.show()
        return None
    if nargout == 1:
        raise ValueError('rfmixer() needs to be called with at least two output arguments.\n')
    if nargout == 2:
        return tOut, signalOut
    raise ValueError('rfmixer() cannot return %d output arguments.' % nargout)
