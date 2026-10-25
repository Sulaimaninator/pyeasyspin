"""Simulation of/compensation for the effect of the resonator"""
import math
import numpy as np
from .._registry import F, nout
from .._struct import Struct, isfield
from .._mat import isempty, vec
from .._matlab import (colon as _colon, interp1 as _interp1, cumtrapz as _cumtrapz,  # noqa: F401
                       round_half_away as _round, fminsearch as _fminsearch)

_EPS = np.finfo(float).eps


@nout(2)
def resonator(t0=None, signal0=None, mwFreq=None, *varargin):
    """resonator(t0, signal0, mwFreq, nu0, QL, option[, Opt]) or
    resonator(t0, signal0, mwFreq, freq, H, option[, Opt]) -> (t, signal).
    option 'simulate' or 'compensate'; t in µs, frequencies in GHz. Outputs are 1 x N rows."""
    if t0 is None:
        return None
    nargin = 3 + len(varargin)
    if nargin < 6:
        raise ValueError('resonator() requires at least 6 input arguments.')
    signal0 = np.asarray(signal0)
    dim = signal0.shape if signal0.ndim >= 2 else (1, signal0.size)
    if min(dim) > 1:
        raise ValueError('Input must be a row or column vector.')
    if len(dim) > 2:
        raise ValueError('Input must be a row or column vector.')
    if np.size(t0) != signal0.size:
        raise ValueError('The time axis and input signal need to be vectors of the same length.')
    mwFreq = float(np.ravel(mwFreq)[0])

    if np.size(varargin[0]) == 1 and np.size(varargin[1]) == 1:
        f, H = _transferfunction('ideal', varargin[0], varargin[1])
    else:
        if np.size(varargin[0]) != np.size(varargin[1]):
            raise ValueError('The frequency axis and resonator transfer function need to have the same length.')
        f, H = _transferfunction('experimental', varargin[0], varargin[1])
    option = varargin[2]
    Opt = Struct(varargin[3]) if nargin == 7 else Struct()

    if not isfield(Opt, 'CutoffFactor') or isempty(Opt.CutoffFactor):
        Opt.CutoffFactor = 1 / 1000
    if not isfield(Opt, 'OverSampleFactor') or isempty(Opt.OverSampleFactor):
        Opt.OverSampleFactor = 10
    estimateTimeStep = not isfield(Opt, 'TimeStep') or isempty(Opt.TimeStep)
    if not isfield(Opt, 'N') or isempty(Opt.N):
        Opt.N = 20
    if not isfield(Opt, 'Window') or isempty(Opt.Window):
        Opt.Window = 'gau'
    if not isfield(Opt, 'alpha') or isempty(Opt.alpha):
        Opt.alpha = 0.6

    t0v = vec(np.asarray(t0, dtype=float))
    s0v = vec(signal0)
    isreal0 = not np.iscomplexobj(signal0)

    dt = (t0v[1] - t0v[0]) / 4
    signal_ = vec(_interp1(t0v, s0v, _colon(0, dt, t0v[-1]), 'spline'))
    N = int(_round((f.size - signal_.size) / 2))
    signal_ = np.concatenate((np.zeros(max(N - 1, 0)), signal_, np.zeros(max(N, 0))))
    FT = np.fft.ifftshift(np.fft.fft(np.fft.fftshift(signal_)))
    f_ = vec(F.fdaxis(dt, FT.size))
    aFT = np.abs(FT)

    # indices below are 1-based, as in EasySpin (including its off-by-one offsets)
    if isreal0 and mwFreq == 0:
        ind0 = int(np.argmin(np.abs(f_))) + 1
        intg = _cumtrapz(aFT[ind0 - 1:])
        indmax = int(np.argmin(np.abs(intg - 0.5 * np.max(intg)))) + 1
        indmax = ind0 + indmax
        indbw = int(np.flatnonzero(aFT[indmax - 1:] > 0.01 * np.max(aFT))[-1]) + 1
    else:
        intg = _cumtrapz(aFT)
        indmax = int(np.argmin(np.abs(intg - 0.5 * np.max(intg)))) + 1
        indbw = int(np.flatnonzero(aFT[indmax - 1:] > 0.01 * np.max(aFT))[-1]) + 1
    startind = indmax - Opt.N * indbw
    endind = indmax + Opt.N * indbw
    delta = max(max(endind - f_.size, 1 - startind), 0)
    i1, i2 = int(startind + delta), int(endind - delta)
    sl = slice(i1 - 1, i2)
    f_pulse = f_[sl]
    FT_pulse = FT[sl]

    H_ = _interp1(f - mwFreq * 1e3, H, f_pulse, 'pchip')
    if option == 'simulate':
        FTc_pulse = FT_pulse * H_
    elif option == 'compensate':
        FTc_pulse = FT_pulse / H_
    else:
        raise ValueError(f"Unknown option '{option}'.")

    if Opt.Window in ('gau', 'exp', 'kai'):
        windowfunction = F.apowin(Opt.Window, FTc_pulse.size, Opt.alpha)
    else:
        windowfunction = F.apowin(Opt.Window, FTc_pulse.size)
    FTc_pulse = FTc_pulse * vec(windowfunction)

    FTc = np.zeros(FT.shape, dtype=complex)
    FTc[sl] = FTc_pulse

    t_ = _colon(-0.5 * FTc.size, 1, 0.5 * FTc.size - 1) * dt
    signal_ = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(FTc)))
    asig = np.abs(signal_)
    if option == 'simulate':
        endind = int(np.flatnonzero(asig > (Opt.CutoffFactor * np.max(asig)))[-1]) + 1
        startind = int(np.argmin(np.abs(t_ + t0v[-1] / 2))) + 1
        t_ = t_[startind - 1:endind] - t_[startind - 1]
        signal_ = signal_[startind - 1:endind]
    else:
        ind = np.flatnonzero(asig > (Opt.CutoffFactor * np.max(asig)))
        t_ = t_[ind[0]:ind[-1] + 1] - t_[ind[0]]
        signal_ = signal_[ind[0]:ind[-1] + 1]

    if estimateTimeStep:
        aFTc = np.abs(FTc)
        indmax = int(np.argmax(aFTc)) + 1
        maxvalue = aFTc[indmax - 1]
        indbw = int(np.flatnonzero(aFTc[indmax - 1:] > 0.05 * maxvalue)[-1]) + 1
        maxFreq = 2 * (f_[indmax + indbw - 2] - f_[indmax - 1])
        Nyquist_dt = 1 / (2 * maxFreq)
        newTimeStep = Nyquist_dt / Opt.OverSampleFactor
        inputTimeStep = t0v[1] - t0v[0]
        Opt.TimeStep = newTimeStep if newTimeStep < inputTimeStep else inputTimeStep

    t = _colon(0, Opt.TimeStep, t_[-1])
    signal = _interp1(t_, signal_, t, 'spline')
    if isreal0 and mwFreq == 0:
        signal = 2 * np.real(signal)
    return t.reshape(1, -1), signal.reshape(1, -1)


def _Hideal(f, f0, Q, nu_max):
    return nu_max * np.ravel(F.resonatorprofile(np.asarray(f).reshape(1, -1), f0, Q, 'transferfunction',
                                                nargout=1))


def _transferfunction(type_, *varargin):
    if type_ == 'ideal':
        f0 = float(np.ravel(varargin[0])[0])
        QL = float(np.ravel(varargin[1])[0])
        fmax = 2 * f0
        N = 2 ** 18
        df = fmax / N
        f = _colon(0, 1, N - 1) * df
        H = _Hideal(f, f0, QL, 1)
        H[0] = H[1]
    else:
        FrequencyAxis = vec(np.asarray(varargin[0], dtype=float))
        FrequencyResponse = vec(np.asarray(varargin[1]))
        fmax = 2 * math.ceil(np.max(FrequencyAxis))
        N = 2 ** 14
        df = fmax / N
        f = _colon(0, 1, N - 1) * df
        ind = np.flatnonzero((f > FrequencyAxis[0]) & (f < FrequencyAxis[-1])) + 1   # 1-based
        FR_ = _interp1(FrequencyAxis, FrequencyResponse, f[ind - 1], 'spline')
        FR_r = FR_i = None
        if np.iscomplexobj(FR_):
            FR_r = np.real(FR_)
            FR_i = np.imag(FR_)
            FR_ = np.abs(FR_)
        maxind = int(np.argmax(FR_)) + 1
        v1max = FR_[maxind - 1]
        f0 = f[ind[0] + maxind - 1]
        v1_3dB = (3 / 4) * v1max
        above = np.flatnonzero(FR_ > v1_3dB) + 1
        ind_3dB = np.array([above[0], above[-1]])
        f_3dB = f[ind[0] - 1 + ind_3dB - 1]
        BW_3dB = np.diff(f_3dB)[0]
        QL = f0 / BW_3dB

        fi = f[ind - 1]
        FRn = FR_.size

        def fitfunc(x):
            return math.sqrt(np.sum((np.abs(_Hideal(fi, x[0], x[1], v1max)) - FR_) ** 2) / FRn)
        x = _fminsearch(fitfunc, np.array([f0, QL]))
        f0, QL = x[0], x[1]
        Hid = _Hideal(f, f0, QL, v1max)
        Hid[0] = Hid[1]

        while FR_[0] - abs(Hid[ind[0] - 1]) < -0.05 * v1max:
            FR_ = FR_[1:]
            ind = ind[1:]
        while FR_[-1] - abs(Hid[ind[-1] - 1]) < -0.05 * v1max:
            FR_ = FR_[:-1]
            ind = ind[:-1]
        a, b = ind[0], ind[-1]
        H0 = np.zeros(f.size)
        H0[:a] = (FR_[0] - abs(Hid[a - 1])) * np.exp(1 / math.log(2) * (f[:a] - f[a - 1])) + np.abs(Hid[:a])
        H0[ind - 1] = FR_
        H0[b - 1:] = (FR_[-1] - abs(Hid[b - 1])) * np.exp(-1 / math.log(2) * (f[b - 1:] - f[b - 1])) + \
            np.abs(Hid[b - 1:])
        with np.errstate(divide='ignore', invalid='ignore'):
            betaid = np.arctan(np.imag(Hid) / np.real(Hid))
        if FR_i is not None:
            betaexp = np.zeros(b)
            with np.errstate(divide='ignore', invalid='ignore'):
                betaexp[ind - 1] = np.arctan(FR_i / FR_r)
        else:
            betaexp = -np.imag(vec(F.hilberttrans(np.log(np.abs(H0)).reshape(1, -1))))
        bi = betaexp[ind - 1]
        den = ind[-1] - ind[0]

        def fitfunc2(x):
            return math.sqrt(np.sum(((bi - x[0]) - betaid[ind - 1]) ** 2) / den)
        x = _fminsearch(fitfunc2, np.array([0.0]))[0]
        beta = np.zeros(f.size)
        beta[ind - 1] = bi - x
        beta[:a] = ((betaexp[a - 1] - x) - betaid[a - 1]) * np.exp(1 / math.log(2) * (f[:a] - f[a - 1])) + betaid[:a]
        beta[b - 1:] = ((betaexp[b - 1] - x) - betaid[b - 1]) * np.exp(-1 / math.log(2) * (f[b - 1:] - f[b - 1])) + \
            betaid[b - 1:]
        H = np.abs(H0) * np.exp(1j * beta)
        H = np.real(H) - 1j * np.imag(vec(F.hilberttrans(np.real(H).reshape(1, -1))))
    H = H / np.max(np.real(H))
    f = f * 1e3
    return f, H
