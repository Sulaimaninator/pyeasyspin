"""Pulse definition"""
import math
import warnings
import numpy as np
from .._registry import F
from .._struct import Struct, isfield
from .._mat import isempty, vec

from .._matlab import (EPS as _EPS, round_half_away as _round, colon as _colon,   # noqa: F401
                       linspace as _linspace, nextpow2 as _nextpow2, cumtrapz as _cumtrapz,
                       trapz as _trapz, narrow as _narrow, cmax_index as _cmax_index,
                       interp1 as _interp1)


# ------------------------------------------------------------------ coefficient databases
# Verbatim copies of easyspin/private/GaussianCascadeCoefficients.txt and
# FourierSeriesCoefficients.txt (EasySpin 6.0.12; the en dashes in the page ranges are '-').
_GAUSSIAN_CASCADE_TXT = """%======================================================================
% Database of parameters for Gaussian pulse cascades
%======================================================================
% Contains the amplitudes, positions and FWHM of Gaussians
% used in Gaussian pulse cascade amplitude-modulated pulses reported
% in the literature.
% Parameters given in following order: positions, amplitudes,
% FWHM in microseconds
%
%----------------------------------------------------------------------
% G3
% Emsley, L., Bodenhausen, G. Gaussian pulse cascades: New analytical
% functions for rectangular selective inversion and in-phase excitation
% in NMR. Chem. Phys. Lett. 165, 469-476 (1990).
% DOI: 10.1016/0009-2614(90)87025-M
% Table 1 on p. 473, Cycle 3
%----------------------------------------------------------------------
0.287 0.508 0.795
-1 1.37 0.49
0.189 0.183 0.243
%----------------------------------------------------------------------
% G4
% Emsley, L., Bodenhausen, G. Gaussian pulse cascades: New analytical
% functions for rectangular selective inversion and in-phase excitation
% in NMR. Chem. Phys. Lett. 165, 469-476 (1990).
% DOI: 10.1016/0009-2614(90)87025-M
% Table 1 on p. 474, Cycle 11
%----------------------------------------------------------------------
0.177 0.492 0.653 0.892
0.62 0.72 -0.91 -0.33
0.172 0.129 0.119 0.139
%----------------------------------------------------------------------
% Q3
% Emsley, L., Bodenhausen, G. Optimization of shaped selective pulses
% for NMR using a quaternion description of their overall propagators.
% J. Magn. Reson. 97, 135-148 (1992).
% DOI: 10.1016/0022-2364(92)90242-Y
% Table 2 on p. 142
%----------------------------------------------------------------------
0.306 0.545 0.804
-4.39 4.57 2.60
0.180 0.183 0.245
%----------------------------------------------------------------------
% Q5
% Emsley, L., Bodenhausen, G. Optimization of shaped selective pulses
% for NMR using a quaternion description of their overall propagators.
% J. Magn. Reson. 97, 135-148 (1992).
% DOI: 10.1016/0022-2364(92)90242-Y
% Table 2 on p. 142
%----------------------------------------------------------------------
0.162 0.307 0.497 0.525 0.803
-1.48 -4.34 7.33 -2.30 5.66
0.186 0.139 0.143 0.290 0.137
%
% end
"""

_FOURIER_SERIES_TXT = """%==============================================================
% Database of coefficients for finite Fourier series pulses
%==============================================================
% Contains the Fourier coefficients used to define selective
% amplitude-modulated pulses reported in the literature
%
%--------------------------------------------------------------
% I-BURP 1
% Geen, H., Freeman, R. Band-selective radiofrequency pulses.
% J. Magn. Reson. 93, 93-141 (1991).
% DOI: 10.1016/0022-2364(91)90034-Q
% Table 5 on p. 117, Np = 256
%--------------------------------------------------------------
0.5
0.70 -0.15 -0.94 0.11 -0.02 -0.04 0.01 -0.02 -0.01
-1.54 1.01 -0.24 -0.04 0.08 -0.04 -0.01 0.01 -0.01
%--------------------------------------------------------------
% I-BURP 2
% Geen, H., Freeman, R. Band-selective radiofrequency pulses.
% J. Magn. Reson. 93, 93-141 (1991).
% DOI: 10.1016/0022-2364(91)90034-Q
% Table 6 on p. 119, Np = 256
%--------------------------------------------------------------
0.5
0.81 0.07 -1.25 -0.24 0.07 0.11 0.05 -0.02 -0.03 -0.02 0.00
-0.68 -1.38 0.20 0.45 0.23 0.05 -0.04 -0.04 0.00 0.01 0.01
%--------------------------------------------------------------
% E-BURP 1
% Geen, H., Freeman, R. Band-selective radiofrequency pulses.
% J. Magn. Reson. 93, 93-141 (1991).
% DOI: 10.1016/0022-2364(91)90034-Q
% Table 3 on p. 112, Np = 256
%--------------------------------------------------------------
0.23
0.88 -1.04 -0.24 0.14 0.03 0.04 -0.03 0.00
-0.40 -1.42 0.77 0.06 0.03 -0.04 -0.02 0.01
%--------------------------------------------------------------
% E-BURP 2
% Geen, H., Freeman, R. Band-selective radiofrequency pulses.
% J. Magn. Reson. 93, 93-141 (1991).
% DOI: 10.1016/0022-2364(91)90034-Q
% Table 4 on p. 115, Np = 256
%--------------------------------------------------------------
0.26
0.91 0.29 -1.28 -0.05 0.04 0.02 0.06 0.00 -0.02 0.00
-0.16 -1.82 0.18 0.42 0.07 0.07 -0.01 -0.04 0.00 0.00
%--------------------------------------------------------------
% U-BURP
% Geen, H., Freeman, R. Band-selective radiofrequency pulses.
% J. Magn. Reson. 93, 93-141 (1991).
% DOI: 10.1016/0022-2364(91)90034-Q
% Table 7 on p. 122, Np = 256
%--------------------------------------------------------------
0.27
-1.42 -0.37 -1.84 4.40 -1.19 0.00 -0.37 0.50 -0.31 0.18 -0.21 0.23 -0.12 0.07 -0.06 0.06 -0.04 0.03 -0.02 0.02
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
%--------------------------------------------------------------
% RE-BURP
% Geen, H., Freeman, R. Band-selective radiofrequency pulses.
% J. Magn. Reson. 93, 93-141 (1991).
% DOI: 10.1016/0022-2364(91)90034-Q
% Table 8 on p. 124, Np = 256
%--------------------------------------------------------------
0.49
-1.02 1.11 -1.57 0.83 -0.42 0.26 -0.16 0.10 -0.07 0.04 -0.03 0.01 -0.02 0.00 -0.01
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
%--------------------------------------------------------------
% SNOB i2
% Kupce, E., Boyd, J., Campbell, I. D. Short Selective Pulses for
% Biochemical Applications. J. Magn. Reson. B 106, 300-303 (1995).
% DOI: 10.1006/jmrb.1995.1049
% Table 1 on p. 300
%--------------------------------------------------------------
0.5
-0.2687 -0.2972 0.0989 -0.0010 -0.0168 0.0009 -0.0017 -0.0013 -0.0014
-1.1461 0.4016 0.0736 -0.0307 0.0079 0.0062 0.0003 -0.0002 0.0009
%--------------------------------------------------------------
% SNOB i3
% Kupce, E., Boyd, J., Campbell, I. D. Short Selective Pulses for
% Biochemical Applications. J. Magn. Reson. B 106, 300-303 (1995).
% DOI: 10.1006/jmrb.1995.1049
% Table 1 on p. 300
%--------------------------------------------------------------
0.5
0.2801 -0.9995 0.1928 0.0967 -0.0480 -0.0148 0.0088 -0.0002 -0.0030
-1.1990 0.4893 0.2439 -0.0816 -0.0409 0.0234 0.0036 -0.0042 0.0001
%
% end
"""


def _sscanf_f(s):
    """sscanf(s,'%f') -> column vector (n x 1); stops at the first non-number."""
    vals = []
    for tok in s.split():
        try:
            vals.append(float(tok))
        except ValueError:
            break
    return np.array(vals, dtype=float).reshape(-1, 1)


def _read_database(text, name, nskip, fields):
    """EasySpin's fgetl loop over a coefficient file: on a line containing name (lower case),
    skip nskip lines and read one '%f' line per field; stop at the first line containing 'end'."""
    lines = text.split('\n')
    pos = 0
    out = {}

    def fgetl():
        nonlocal pos
        s = lines[pos] if pos < len(lines) else ''
        pos += 1
        return s

    while True:
        s = fgetl()
        if name in s.lower():
            for _ in range(nskip):
                fgetl()
            for fld in fields:
                s = fgetl()
                out[fld] = _sscanf_f(s)
        if 'end' in s:
            break
        if pos > len(lines) + 1:
            break
    return out


# ------------------------------------------------------------------ pulse
def _num(x):
    return np.asarray(x, dtype=float).ravel(order='F') if not np.isscalar(x) else np.array([float(x)])


def _sech(x):
    return 1.0 / np.cosh(x)


_FS_NAMES = ('fourierseries', 'i-burp 1', 'i-burp 2', 'snob i2', 'snob i3', 'e-burp 1', 'e-burp 2', 'u-burp',
             're-burp')
_GC_NAMES = ('gaussiancascade', 'g3', 'g4', 'q3', 'q5')


def _scal(x):
    return float(np.ravel(x)[0])


def pulse(*varargin, nargout=None, plot=False):
    """pulse(Par[, Opt], nargout=2) -> (t, IQ) or, with nargout=3, (t, IQ, modulation).
    t in microseconds (1xN row), IQ in MHz. plot=True draws the shape (MATLAB nargout=0)."""
    if nargout is None:
        nargout = 0 if plot else 2
    if len(varargin) == 0:
        return None
    if len(varargin) == 1:
        Par = varargin[0]
        Opt = Struct()
    elif len(varargin) == 2:
        Par, Opt = varargin
    else:
        raise ValueError('pulse() needs 1 or 2 input arguments.')
    if not isinstance(Par, dict):
        raise ValueError('First input argument must be a structure.')
    if not isinstance(Opt, dict):
        raise ValueError('Second input argument must be a structure.')
    Par = Struct(Par)
    Opt = Struct(Opt)
    plotResults = nargout == 0

    if not isfield(Par, 'tp'):
        raise ValueError('Par.tp (pulse length, in µs) is missing.')
    Par.tp = _scal(Par.tp)

    userIQ = ((isfield(Par, 'IQ') and not isempty(Par.IQ)) or (isfield(Par, 'I') and not isempty(Par.I))
              or (isfield(Par, 'Q') and not isempty(Par.Q)))

    if not isfield(Par, 'Flip') or isempty(Par.Flip):
        if not isfield(Par, 'Amplitude') and not isfield(Par, 'Qcrit'):
            if not userIQ:
                raise ValueError('Par.Flip (pulse flip angle, in radians) is missing.\n'
                                 'Give Par.Flip, or provide Par.Amplitude or Par.Qcrit.')
            Par.Amplitude = np.zeros((0, 0))
        else:
            Par.Flip = np.zeros((0, 0))

    if not isfield(Par, 'Frequency') or isempty(Par.Frequency):
        Par.Frequency = 0.0
    if not isfield(Par, 'Phase'):
        Par.Phase = 0.0

    estimateTimeStep = not isfield(Par, 'TimeStep') or isempty(Par.TimeStep)

    if not isfield(Opt, 'OverSampleFactor'):
        Opt.OverSampleFactor = 10.0
    calculateExciteProfile = plotResults

    BWCompensation = isfield(Par, 'FrequencyResponse') or isfield(Par, 'ResonatorFrequency') or \
        isfield(Par, 'ResonatorQL')
    if BWCompensation:
        if isfield(Par, 'FrequencyResponse') and not isempty(Par.FrequencyResponse):
            FR = np.atleast_2d(np.asarray(Par.FrequencyResponse))
            n1, n2 = FR.shape
            if n2 == 2:
                FR = FR.T
            elif n1 != 2 and n2 != 2:
                raise ValueError('Par.FrequencyResponse should contain both the frequency axis and the '
                                 'resonator transfer function.')
            Par.FrequencyResponse = FR
        if isfield(Par, 'ResonatorFrequency') and not isempty(Par.ResonatorFrequency):
            if not isfield(Par, 'ResonatorQL') or isempty(Par.ResonatorQL):
                raise ValueError('Use of the resonator bandwidth compensation with an ideal transfer '
                                 'function requires the inputs Par.ResonatorFrequency and Par.ResonatorQL.')
        if not isfield(Par, 'mwFreq') or isempty(Par.mwFreq):
            raise ValueError('Par.mwFreq is required for the resonator bandwidth compensation.')

    modulation = Struct()
    FrequencyModulation = 'none'
    if userIQ:
        if not isfield(Par, 'Type') or isempty(Par.Type):
            Par.Type = 'user-IQ'
        if isfield(Par, 'IQ'):
            Par.I = np.real(Par.IQ)
            Par.Q = np.imag(Par.IQ)
        elif not isfield(Par, 'I'):
            Par.Q = np.atleast_2d(np.asarray(Par.Q, dtype=float))
            Par.I = np.zeros(Par.Q.shape)
        elif not isfield(Par, 'Q'):
            Par.I = np.atleast_2d(np.asarray(Par.I, dtype=float))
            Par.Q = np.zeros(Par.I.shape)
        elif np.all(np.array(np.atleast_2d(Par.I).shape) != np.array(np.atleast_2d(Par.Q).shape)):
            raise ValueError('I and Q input data have different lengths.')
        I2, Q2 = np.atleast_2d(np.asarray(Par.I)), np.atleast_2d(np.asarray(Par.Q))
        if I2.ndim != 2 or min(I2.shape) != 1 or Q2.ndim != 2 or min(Q2.shape) != 1:
            raise ValueError('I and Q input data should be vectors.')
        t = _linspace(0, Par.tp, I2.size)
        IQ = (I2.astype(float) + 1j * Q2.astype(float)).reshape(I2.shape)
        if isfield(Par, 'TimeStep') and not isempty(Par.TimeStep):
            t0 = t
            t = _colon(t0[0], _scal(Par.TimeStep), t0[-1])
            IQ = _interp1(t0, IQ, t, 'spline').reshape(1, -1)
        else:
            Par.TimeStep = t[1] - t[0]
        if not isempty(Par.Amplitude):
            IQ = _narrow(_scal(Par.Amplitude) * IQ)
        modulation.A = np.zeros((0, 0))
        modulation.freq = np.zeros((0, 0))
        modulation.phase = np.zeros((0, 0))
        t = t.reshape(1, -1)
    else:
        if not isfield(Par, 'Type') or isempty(Par.Type):
            raise ValueError("Par.Type (pulse shape) is missing. Example: Par.Type = 'sech/tanh'.")
        shape = Par.Type.split('/')
        if len(shape) == 1:
            AmplitudeModulation = [s.lower() for s in shape]
            FrequencyModulation = 'none'
        elif len(shape) == 2:
            AmplitudeModulation = [s.lower() for s in shape[0].split('*')]
            FrequencyModulation = shape[1].lower() if shape[1] else 'none'
        else:
            raise ValueError("Par.Type must contain the pulse shape in the form 'AM/FM'.")

        for am in AmplitudeModulation:
            if am == 'rectangular':
                pass
            elif am == 'gaussian':
                if (not isfield(Par, 'tFWHM') or isempty(Par.tFWHM)) and \
                        (not isfield(Par, 'trunc') or isempty(Par.trunc)):
                    raise ValueError("Amplitude modulation function 'gaussian' not sufficiently defined. "
                                     'Specify Par.tFWHM or Par.trunc.')
                elif not isfield(Par, 'tFWHM') and isfield(Par, 'trunc') and not isempty(Par.trunc):
                    Par.tFWHM = math.sqrt(-(Par.tp ** 2) / math.log2(_scal(Par.trunc)))
                    if Par.tFWHM == 0:
                        Par.tFWHM = _scal(Par.TimeStep) / 2
            elif am == 'sinc':
                if not isfield(Par, 'zerocross') or isempty(Par.zerocross):
                    raise ValueError("Amplitude modulation function 'sinc' not sufficiently defined. "
                                     'Specify Par.zerocross.')
            elif am == 'halfsin':
                pass
            elif am == 'quartersin':
                if not isfield(Par, 'trise') or isempty(Par.trise):
                    raise ValueError("Amplitude modulation function 'quartersin' not sufficiently defined. "
                                     'Specify Par.trise.')
            elif am == 'tanh2':
                if not isfield(Par, 'trise') or isempty(Par.trise):
                    raise ValueError("Amplitude modulation function 'tanh2' not sufficiently defined. "
                                     'Specify Par.trise.')
            elif am == 'sech':
                if not isfield(Par, 'beta') or isempty(Par.beta):
                    raise ValueError("Amplitude modulation function 'sech' not sufficiently defined. "
                                     'Specify Par.beta parameter (in 1/µs).')
                if not isfield(Par, 'n') or isempty(Par.n):
                    Par.n = 1.0
                nn = np.asarray(Par.n)
                if nn.dtype.kind not in 'iufc' or nn.size not in (1, 2):
                    raise ValueError('Par.n must contain 1 or 2 numbers.')
                if np.iscomplexobj(nn) or np.any(np.mod(nn, 1)) or np.any(nn < 1):
                    raise ValueError('Par.n must contain nonnegative integers.')
            elif am == 'wurst':
                if not isfield(Par, 'nwurst') or isempty(Par.nwurst):
                    raise ValueError('Amplitude modulation function ``WURST`` not sufficiently defined. '
                                     'Specify Par.nwurst.')
                if np.size(Par.nwurst) != 1 or np.mod(_scal(Par.nwurst), 1) or _scal(Par.nwurst) < 1:
                    raise ValueError('Pulseshape.nwurst must be a nonnegative integer (1,2,...).')
            elif am in _GC_NAMES:
                if not (isfield(Par, 'A0') and isfield(Par, 'x0') and isfield(Par, 'FWHM')):
                    if am == 'gaussiancascade':
                        raise ValueError('The amplitudes A0, positions x0 and FWHM of the '
                                         'Gaussians are required as input.')
                    else:
                        d = _read_database(_GAUSSIAN_CASCADE_TXT, am, 6, ('x0', 'A0', 'FWHM'))
                        Par.update(d)
                elif not (np.size(Par.A0) == np.size(Par.x0) and np.size(Par.A0) == np.size(Par.FWHM)):
                    raise ValueError('The same number of parameters is required for the A0, '
                                     'x0 and FWHM inputs.')
            elif am in _FS_NAMES:
                if (not (isfield(Par, 'A0') and isfield(Par, 'An') and isfield(Par, 'Bn'))
                        or (isempty(Par.A0) and isempty(Par.An) and isempty(Par.Bn))):
                    if am == 'fourierseries':
                        raise ValueError('The Fourier coefficients A0, An and Bn are required as input.')
                    else:
                        d = _read_database(_FOURIER_SERIES_TXT, am, 5, ('A0', 'An', 'Bn'))
                        Par.update(d)
                elif np.size(Par.An) != np.size(Par.Bn):
                    raise ValueError('The same number of A and B Fourier coefficients is required.')
            else:
                raise ValueError(f"The amplitude modulation function '{am}' is not defined.")

        needsFreqRange = FrequencyModulation != 'none'
        Fr = _num(Par.Frequency)
        if needsFreqRange:
            if Fr.size != 2 or np.diff(Fr)[0] == 0:
                raise ValueError(f"Frequency modulation function '{FrequencyModulation}' not sufficiently "
                                 'defined.\nSpecify frequency range in Par.Frequency (in MHz).')
        else:
            if Fr.size > 1:
                if Fr[0] == Fr[1]:
                    Par.Frequency = float(Fr[0])
                else:
                    raise ValueError("Frequency modulation is set to 'none', but a frequency "
                                     'range is given in Par.Frequency. Please define a single '
                                     'pulse frequency in Par.Frequency (in MHz).')
        Fr = _num(Par.Frequency)

        if FrequencyModulation in ('none', 'linear', 'uniformq'):
            pass
        elif FrequencyModulation == 'tanh':
            if not isfield(Par, 'beta') or isempty(Par.beta):
                raise ValueError("Frequency modulation function 'tanh' not sufficiently defined. "
                                 'Specify dimensionless Par.beta parameter.')
        else:
            raise ValueError(f"The frequency modulation function '{FrequencyModulation}' is not defined.")

        if 'sech' in AmplitudeModulation and FrequencyModulation == 'tanh' and \
                (isfield(Par, 'n') and not isempty(Par.n) and np.any(np.asarray(Par.n) != 1)):
            warnings.warn('For uniform-adiabaticity pulses with %dth-order sech amplitude'
                          "modulation use Par.Type = 'sech/uniformQ'." % int(np.max(Par.n)))

        if isfield(Par, 'Qcrit') and not isempty(Par.Qcrit):
            if FrequencyModulation == 'none':
                warnings.warn('For a pulse without frequency modulation, the Qcrit input '
                              'parameter is ignored.')
                Par.Qcrit = np.zeros((0, 0))
                if not isfield(Par, 'Amplitude') or isempty(Par.Amplitude):
                    Par.Flip = math.pi
            else:
                Par.Amplitude = np.zeros((0, 0))
                Par.Flip = np.zeros((0, 0))

        tp = Par.tp
        if estimateTimeStep:
            if FrequencyModulation == 'none':
                FM_BW = 0.0
            else:
                FM_BW = abs(Fr[1] - Fr[0])
            dt = 0.1e-3
            t0 = _colon(0, dt, tp)
            ti0 = t0 - 0.5 * tp
            A0 = np.ones(t0.size)
            for am in AmplitudeModulation:
                if am == 'rectangular':
                    A0 = A0 * np.ones(t0.size)
                elif am == 'gaussian':
                    A0 = A0 * np.exp(-(4 * math.log(2) * ti0 ** 2) / _scal(Par.tFWHM) ** 2)
                elif am == 'sinc':
                    x_ = (2 * math.pi * ti0) / _scal(Par.zerocross)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        A1 = np.sin(x_) / x_
                    A1[np.isnan(A1)] = 1
                    A0 = A0 * (A1 / np.max(A1))
                elif am == 'halfsin':
                    A1 = np.cos(math.pi * ti0 / tp)
                    A0 = A0 * A1
                elif am == 'quartersin':
                    A1 = np.ones(t0.size)
                    trise = _scal(Par.trise)
                    if trise != 0 and 2 * trise < tp:
                        tpartial = _colon(0, dt, trise)
                        npts = tpartial.size
                        A1[:npts] = np.sin(tpartial * (math.pi / (2 * trise)))
                        A1[A1.size - npts:] = A1[npts - 1::-1]
                    A0 = A0 * A1
                elif am == 'tanh2':
                    trise = _scal(Par.trise)
                    A1 = (1 / math.tanh(tp / 2 / trise)) ** 4 * np.tanh(t0 / trise) ** 2 * \
                        np.tanh((tp - t0) / trise) ** 2
                    A0 = A0 * A1
                elif am == 'sech':
                    n = float(np.min(Par.n))
                    A0 = A0 * _sech(_scal(Par.beta) * 2 ** (n - 1) * (ti0 / tp) ** n)
                elif am == 'wurst':
                    A0 = A0 * (1 - np.abs(np.sin(math.pi * ti0 / tp)) ** _scal(Par.nwurst))
                elif am in _GC_NAMES:
                    A0 = np.zeros(t0.size)
                    pA0, px0, pF = _num(Par.A0), _num(Par.x0), _num(Par.FWHM)
                    for j in range(pA0.size):
                        A0 = A0 + pA0[j] * np.exp(-(4 * math.log(2) / (pF[j] * tp) ** 2) * (t0 - px0[j] * tp) ** 2)
                    A0 = A0 / np.max(A0)
                elif am in _FS_NAMES:
                    pAn, pBn = _num(Par.An), _num(Par.Bn)
                    A0 = np.zeros(t0.size) + _scal(Par.A0)
                    for j in range(pAn.size):
                        jj = j + 1
                        A0 = A0 + pAn[j] * np.cos(jj * 2 * math.pi * t0 / tp) + pBn[j] * np.sin(jj * 2 * math.pi * t0 / tp)
                    A0 = A0 / np.max(A0)
            if _nextpow2(t0.size) < 10:
                zf = 2 ** 10
            else:
                zf = 4 * 2 ** _nextpow2(t0.size)
            A0ft = np.abs(np.fft.fftshift(np.fft.fft(A0, zf)))
            f = np.ravel(F.fdaxis(dt, zf))
            intg = _cumtrapz(A0ft)
            indmax = int(np.argmin(np.abs(intg - 0.5 * np.max(intg)))) + 1      # 1-based
            indbw = np.flatnonzero(A0ft[indmax - 1:] > 0.1 * np.max(A0ft)) + 1   # 1-based
            AM_BW = 2 * (f[indmax + indbw[-1] - 1] - f[indmax - 1])
            BW = max(FM_BW, AM_BW)
            maxFreq = max(abs(np.mean(Fr) + np.array([-1.0, 1.0]) * BW / 2))
            if maxFreq != 0:
                Nyquist_dt = 1 / (2 * maxFreq)
                Par.TimeStep = Nyquist_dt / _scal(Opt.OverSampleFactor)
            else:
                Par.TimeStep = 0.002
            if Par.TimeStep > tp:
                Par.TimeStep = tp
            Par.TimeStep = tp / _round(tp / Par.TimeStep)

        TimeStep = _scal(Par.TimeStep)
        t = _colon(0, TimeStep, tp)
        ti = t - tp / 2
        nPoints = t.size

        modulation.A = np.ones(nPoints)
        for am in AmplitudeModulation:
            if am == 'rectangular':
                A = np.ones(nPoints)
            elif am == 'gaussian':
                A = np.exp(-(4 * math.log(2) * ti ** 2) / _scal(Par.tFWHM) ** 2)
            elif am == 'sinc':
                x_ = 2 * math.pi * ti / _scal(Par.zerocross)
                with np.errstate(invalid='ignore', divide='ignore'):
                    A = np.sin(x_) / x_
                A[np.isnan(A)] = 1
                A = A / np.max(A)
            elif am == 'halfsin':
                A = np.cos(math.pi * ti / tp)
            elif am == 'quartersin':
                A = np.ones(nPoints)
                trise = _scal(Par.trise)
                if trise != 0 and 2 * trise < tp:
                    tpartial = _colon(0, TimeStep, trise)
                    npts = tpartial.size
                    A[:npts] = np.sin(tpartial * (math.pi / (2 * trise)))
                    A[A.size - npts:] = A[npts - 1::-1]
            elif am == 'tanh2':
                trise = _scal(Par.trise)
                A = (1 / math.tanh(tp / 2 / trise)) ** 4 * np.tanh(t / trise) ** 2 * np.tanh((tp - t) / trise) ** 2
            elif am == 'sech':
                beta = _scal(Par.beta)
                pn = _num(Par.n)
                if pn.size == 1:
                    if pn[0] == 1:
                        A = _sech(beta * ti / tp)
                    else:
                        A = _sech(beta * 0.5 * (2 * ti / tp) ** pn[0])
                else:
                    idxL = ti < 0
                    idxR = ~idxL
                    A = np.zeros(nPoints)
                    A[idxL] = _sech(beta * 0.5 * (2 * ti[idxL] / tp) ** pn[0])
                    A[idxR] = _sech(beta * 0.5 * (2 * ti[idxR] / tp) ** pn[1])
            elif am == 'wurst':
                A = 1 - np.abs(np.sin(math.pi * ti / tp)) ** _scal(Par.nwurst)
            elif am in _GC_NAMES:
                A = np.zeros(t.size)
                pA0, px0, pF = _num(Par.A0), _num(Par.x0), _num(Par.FWHM)
                for j in range(pA0.size):
                    A = A + pA0[j] * np.exp(-(4 * math.log(2) / (pF[j] * tp) ** 2) * (t - px0[j] * tp) ** 2)
                A = A / np.max(A)
            elif am in _FS_NAMES:
                pAn, pBn = _num(Par.An), _num(Par.Bn)
                A = np.zeros(t.size) + _scal(Par.A0)
                for j in range(pAn.size):
                    jj = j + 1
                    A = A + pAn[j] * np.cos(jj * 2 * math.pi * t / tp) + pBn[j] * np.sin(jj * 2 * math.pi * t / tp)
                A = A / np.max(A)
            modulation.A = modulation.A * A

        if FrequencyModulation == 'none':
            modulation.freq = np.zeros(nPoints)
            modulation.phase = np.zeros(nPoints)
        elif FrequencyModulation == 'linear':
            k = (Fr[1] - Fr[0]) / tp
            modulation.freq = k * ti
            modulation.phase = 2 * math.pi * ((k / 2) * ti ** 2)
        elif FrequencyModulation == 'tanh':
            beta = _scal(Par.beta)
            Par.BWinf = (Fr[1] - Fr[0]) / math.tanh(beta / 2)
            modulation.freq = (Par.BWinf / 2) * np.tanh((beta / tp) * ti)
            modulation.phase = (Par.BWinf / 2) * (tp / beta) * np.log(np.cosh((beta / tp) * ti))
            modulation.phase = 2 * math.pi * modulation.phase
        elif FrequencyModulation == 'uniformq':
            modulation.freq = _cumtrapz(ti, modulation.A ** 2) / _trapz(ti, modulation.A ** 2)
            modulation.freq = (Fr[1] - Fr[0]) * (modulation.freq - 1 / 2)
            modulation.phase = 2 * math.pi * _cumtrapz(ti, modulation.freq)
            modulation.phase = modulation.phase + abs(np.min(modulation.phase))

        if BWCompensation and FrequencyModulation != 'none':
            nu0 = modulation.freq
            A0 = modulation.A
            newaxis = nu0 + np.mean(Fr) + _scal(Par.mwFreq) * 1e3
            profile = None
            if isfield(Par, 'FrequencyResponse'):
                f = Par.FrequencyResponse[0, :] * 1e3
                H = Par.FrequencyResponse[1, :]
                if np.min(newaxis) < np.min(f) or np.max(newaxis) > np.max(f):
                    raise ValueError('The frequency sweep width of the pulse extends further than '
                                     'the given resonator profile. Please provide the resonator '
                                     'profile for the complete pulse frequency sweep width.')
                if np.iscomplexobj(H):
                    H = np.abs(H)
                profile = _interp1(f, H, newaxis)
            elif isfield(Par, 'ResonatorFrequency'):
                f0 = _scal(Par.ResonatorFrequency) * 1e3
                QL = _scal(Par.ResonatorQL)
                profile = np.abs(np.ravel(F.resonatorprofile(newaxis.reshape(1, -1), f0, QL, 'transferfunction',
                                                            nargout=1)))
            if FrequencyModulation == 'uniformq' or Par.Type == 'sech/tanh':
                profile = A0 * profile
            profile = np.array(profile, dtype=float)
            profile[profile == 0] = 1e-3
            int_ = _cumtrapz(nu0, profile ** -2.0)
            t_f = t[-1] * int_ / int_[-1]
            if t_f.size != np.unique(t_f).size:
                _, ind = np.unique(t_f, return_index=True)
                t_f = t_f[ind]
                nu0 = nu0[ind]
                A0 = A0[ind]
            nu_adapted = _interp1(t_f, nu0, t, 'pchip')
            modulation.freq = nu_adapted
            modulation.phase = 2 * math.pi * _cumtrapz(t, modulation.freq)
            modulation.phase = modulation.phase + abs(np.min(modulation.phase))
            if FrequencyModulation == 'uniformq' or Par.Type == 'sech/tanh':
                modulation.A = _interp1(nu0, A0, nu_adapted, 'pchip')

        if not isfield(Par, 'Amplitude') or isempty(Par.Amplitude):
            if FrequencyModulation == 'none':
                if isfield(Par, 'Flip') and not isempty(Par.Flip):
                    Par.Amplitude = _scal(Par.Flip) / (2 * math.pi * _trapz(t, modulation.A))
            else:
                if (not isfield(Par, 'Qcrit') or isempty(Par.Qcrit)) and \
                        (isfield(Par, 'Flip') and not isempty(Par.Flip)):
                    if _scal(Par.Flip) > math.pi:
                        raise ValueError('Pulse amplitude calculation from flip angle not applicable for '
                                         'angles larger than pi.')
                    with np.errstate(divide='ignore'):   # Flip = pi: 2/0 = Inf -> Qcrit = 5
                        Par.Qcrit = (2 / math.pi) * np.log(np.float64(2) / (1 + math.cos(_scal(Par.Flip))))
                    Par.Qcrit = float(min(Par.Qcrit, 5))
                if not BWCompensation:
                    if FrequencyModulation == 'linear':
                        sweeprate = abs(Fr[1] - Fr[0]) / tp
                    elif FrequencyModulation == 'tanh':
                        sweeprate = _scal(Par.beta) * abs(Par.BWinf) / (2 * tp)
                    elif FrequencyModulation == 'uniformq':
                        ind = int(np.argmin(np.abs(ti)))
                        dnu = np.abs(np.diff(2 * math.pi * modulation.freq / (t[1] - t[0])))
                        sweeprate = dnu[ind] / (2 * math.pi * (modulation.A[ind]) ** 2)
                else:
                    ind = int(np.argmin(np.abs(ti)))
                    dnu = np.abs(np.diff(2 * math.pi * modulation.freq / (t[1] - t[0])))
                    sweeprate = dnu[ind] / (2 * math.pi * (modulation.A[ind]) ** 2)
                Par.Amplitude = math.sqrt(2 * math.pi * _scal(Par.Qcrit) * sweeprate) / (2 * math.pi)

        modulation.A = _scal(Par.Amplitude) * modulation.A
        totalphase = modulation.phase + 2 * math.pi * np.mean(Fr) * t + Par.Phase
        IQ = _narrow(modulation.A * np.exp(1j * totalphase))
        t = t.reshape(1, -1)
        IQ = IQ.reshape(1, -1)
        modulation.A = modulation.A.reshape(1, -1)
        modulation.freq = np.asarray(modulation.freq).reshape(1, -1)
        modulation.phase = np.asarray(modulation.phase).reshape(1, -1)

    exprof = None
    if calculateExciteProfile:
        exprof = F.exciteprofile(t, IQ, nargout=2)

    if plotResults:
        _plot_pulse(t, IQ, modulation, Par, FrequencyModulation, exprof)

    if nargout == 0:
        return None
    if nargout == 2:
        return t, IQ
    if nargout == 3:
        meanF = np.mean(_num(Par.Frequency))
        if not isempty(modulation.freq):
            modulation.freq = modulation.freq + meanF
        if not isempty(modulation.phase):
            modulation.phase = modulation.phase + 2 * math.pi * meanF * t + Par.Phase
        return t, IQ, modulation
    raise ValueError('The function pulse() needs 2 or 3 output arguments.')


def _plot_pulse(t, IQ, modulation, Par, FM, exprof):
    import matplotlib.pyplot as plt
    t = np.ravel(t)
    IQ = np.ravel(IQ)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.suptitle('Type = ' + str(Par.get('Type', '')))
    if not isempty(modulation.A):
        ax[0].plot(t, np.ravel(modulation.A), color='0.9')
        ax[0].plot(t, -np.ravel(modulation.A), color='0.9')
    ax[0].plot(t, np.real(IQ), 'b', label='I')
    ax[0].plot(t, np.imag(IQ), 'r', label='Q')
    ax[0].set_xlabel('t (µs)')
    ax[0].set_ylabel('amplitude (MHz)')
    ax[0].legend()
    if not isempty(modulation.freq):
        ax[1].plot(t, np.ravel(modulation.freq) + np.mean(_num(Par.Frequency)), 'b')
    ax[1].set_xlabel('t (µs)')
    ax[1].set_ylabel('frequency (MHz)')
    if exprof is not None:
        off, M = exprof
        off = np.ravel(off)
        for i, (c, lab) in enumerate(zip(('g', 'r', 'b'), 'xyz')):
            ax[2].plot(off, M[i, :], c, label=lab, visible=(i == 2))
        ax[2].set_xlabel('frequency (MHz)')
        ax[2].set_ylabel('M_i/M_0')
        ax[2].set_ylim(-1, 1)
    plt.show()
