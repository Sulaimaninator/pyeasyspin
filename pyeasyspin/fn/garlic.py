"""Simulates isotropic and fast-motion cw EPR spectra"""
import warnings
import numpy as np
from .._registry import F, nout
from .._struct import Struct, isfield
from .._mat import a2, vec, isempty, numel
from ..config import config


def _c(name):
    v = getattr(F, name)()
    return v[0] if isinstance(v, tuple) else v


def _if(x):
    """MATLAB `if x`: nonempty and all elements nonzero."""
    a = np.asarray(x)
    return bool(a.size) and bool(np.all(a != 0))


def _mround(x):
    """MATLAB round (half away from zero)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _isreal(x):
    return not np.iscomplexobj(x)


def _scalar(x):
    return float(np.ravel(np.asarray(x, dtype=float))[0])


def _linspace(d1, d2, n):
    return np.linspace(float(d1), float(d2), int(n))


def _constructstickspectrum(Positions, Amplitudes, Range, nPoints):
    Positions = vec(Positions)
    Amplitudes = vec(Amplitudes)
    Range = vec(Range)
    idxPositions = (Positions - Range[0]) / (Range[1] - Range[0]) * (nPoints - 1)
    idxPositions = 1 + _mround(idxPositions)
    inRange = (idxPositions >= 1) & (idxPositions <= nPoints)
    if np.any(idxPositions < 1):
        F.logmsg(0, '** Spectrum exceeds sweep range. Artifacts at lower limit possible.')
    if np.any(idxPositions > nPoints):
        F.logmsg(0, '** Spectrum exceeds sweep range. Artifacts at upper limit possible.')
    Spectrum = np.zeros(int(nPoints))
    np.add.at(Spectrum, idxPositions[inRange].astype(int) - 1, Amplitudes[inRange])
    return Spectrum.reshape(1, -1)


@nout(-1)
def garlic(Sys=None, Exp=None, Opt=None, nargout=None, plot=False):
    """garlic(Sys, Exp, Opt) -> (x, spec); nargout=1 -> spec, nargout=3 -> (x, spec, info).
    Isotropic (Breit-Rabi or perturbation) and fast-motion (fastmotion line widths) cw EPR.
    plot=True shows the spectrum via cwepr_plot (MATLAB's nargout=0 behaviour)."""
    if nargout is None:
        nargout = 2
    if Sys is None:
        raise ValueError('Not enough input arguments.')
    if Exp is None:
        raise ValueError('Experimental parameters (2nd input) are missing!')
    if nargout not in (0, 1, 2, 3):
        raise ValueError('Wrong number of output parameters!')
    if isinstance(Sys, dict):
        Sys = Struct(Sys)
    Exp = Struct(Exp) if isinstance(Exp, dict) else Struct()
    Opt = Struct(Opt) if isinstance(Opt, dict) else Struct()

    if not isfield(Opt, 'Verbosity'):
        Opt.Verbosity = 0
    config.LogLevel = Opt.Verbosity

    Exp.FrequencySweep = (not isfield(Exp, 'mwFreq')) and isfield(Exp, 'Field')
    if Exp.FrequencySweep:
        SweepAutoRange = (not isfield(Exp, 'mwRange') or isempty(Exp.mwRange)) and \
            (not isfield(Exp, 'mwCenterSweep') or isempty(Exp.mwCenterSweep))
    else:
        SweepAutoRange = (not isfield(Exp, 'Range') or isempty(Exp.Range)) and \
            (not isfield(Exp, 'CenterSweep') or isempty(Exp.CenterSweep))

    if not isfield(Opt, 'IsoCutoff'):
        Opt.IsoCutoff = 1e-6
    else:
        if Opt.IsoCutoff < 0 or Opt.IsoCutoff > 1:
            raise ValueError('Options.IsoCutoff must be between 0 and 1.')

    if not isfield(Opt, 'separate'):
        Opt.separate = ''
    separateOutput, err = F.parseoption(Opt, 'separate', ['', 'components', 'transitions', 'orientations', 'sites'],
                                        nargout=2)
    if not isempty(err):
        raise ValueError(err)
    separateComponentSpectra = separateOutput == 2
    separateTransitionSpectra = separateOutput == 3
    separateOrientationSpectra = separateOutput == 4
    separateSiteSpectra = separateOutput == 5
    if separateTransitionSpectra or separateSiteSpectra or separateOrientationSpectra:
        separateComponentSpectra = True
    if separateTransitionSpectra:
        raise ValueError("garlic does not support Opt.separate='transitions'.")
    if separateSiteSpectra:
        raise ValueError("garlic does not support Opt.separate='sites'.")
    if separateOrientationSpectra:
        raise ValueError("garlic does not support Opt.separate='orientations'.")

    singleIsotopologue = isfield(Sys, 'singleiso') and _if(Sys.singleiso)
    if not singleIsotopologue:
        thirdOutput = nargout >= 3
        xAxis, spec, info = F.compisoloop(F.garlic, Sys, Exp, Opt, SweepAutoRange, thirdOutput,
                                          separateComponentSpectra)
        if plot or nargout == 0:
            if plot:
                F.cwepr_plot(xAxis, spec, Exp)
            if nargout == 0:
                return None
        if nargout == 1:
            return spec
        if nargout == 2:
            return xAxis, spec
        return xAxis, spec, info

    EasySpinLogLevel = config.LogLevel
    if EasySpinLogLevel is not None and EasySpinLogLevel >= 1:
        import datetime
        F.logmsg(1, '=begin=garlic=====' + datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S') + '=================')

    Sys, err = F.validatespinsys(Sys)
    if not isempty(err):
        raise ValueError(err)
    Sys = Struct(Sys)
    if _if(Sys.MO_present):
        raise ValueError('salt does not support Sys.Ham* parameters!')
    if np.any(a2(Sys.get('L', 0)) != 0):
        raise ValueError('salt does not support Sys.L!')
    if int(Sys.nElectrons) != 1 or (isfield(Sys, 'L') and np.any(a2(Sys.L) != 0)):
        raise ValueError('Only systems with one electron spin S=1/2 are supported.')
    elif np.any(a2(Sys.S) != 0.5):
        raise ValueError('Only systems with electron spin S=1/2 are supported.')
    if isfield(Sys, 'Exchange') and _if(np.asarray(Sys.Exchange) != 0):
        raise ValueError('garlic does not support Heisenberg exchange.')
    if isfield(Sys, 'nn') and np.any(a2(Sys.nn) != 0):
        raise ValueError('garlic does not support nuclear-nuclear couplings (Sys.nn).')

    if isfield(Sys, 'logtcorr'):
        Sys.tcorr = 10.0 ** np.asarray(Sys.logtcorr, dtype=float)
        if Sys.tcorr.size == 1:
            Sys.tcorr = float(Sys.tcorr)
    if not isfield(Sys, 'tcorr'):
        Sys.tcorr = 0
    FastMotionRegime = (not isempty(Sys.tcorr)) and bool(np.all(np.asarray(Sys.tcorr) != 0))
    if FastMotionRegime:
        if not _isreal(Sys.tcorr):
            raise ValueError('Problem with System.tcorr: must be a number!')
        if numel(Sys.tcorr) != 1:
            raise ValueError('Problem with System.tcorr: must be a single number!')
        Sys.tcorr = _scalar(Sys.tcorr)
        if Sys.tcorr <= 0:
            raise ValueError('Problem with System.tcorr: must be positive!')
        if Sys.tcorr < 1e-13:
            raise ValueError('Correlation time too small (%f seconds).' % Sys.tcorr)
        if isfield(Sys, 'n'):
            if np.any(a2(Sys.n) > 1):
                raise ValueError('Cannot treat equivalent nuclei in fast-motion regime!\n Please rewrite spin system!')
        if isfield(Sys, 'Potential'):
            raise ValueError('garlic cannot simulate fast-motion spectra with an orienting potential (Sys.Potential).')

    lw = a2(Sys.lw).astype(float).ravel(order='F').copy()
    lwpp = a2(Sys.lwpp).astype(float).ravel(order='F')
    ConvolutionBroadening = bool(np.any(lw > 0)) or FastMotionRegime

    DefaultExp = Struct()
    DefaultExp.Harmonic = np.zeros((0, 0))
    DefaultExp.nPoints = 1024
    DefaultExp.ModAmp = 0
    DefaultExp.mwMode = 'perpendicular'
    DefaultExp.mwPhase = 0
    DefaultExp.Temperature = np.nan
    Exp = F.adddefaults(Exp, DefaultExp)

    if isfield(Exp, 'Orientations'):
        raise ValueError('Exp.Orientations is no longer supported, use Exp.SampleFrame instead.')
    if isfield(Exp, 'CrystalOrientation'):
        raise ValueError('Exp.CrystalOrientation is no longer supported, use Exp.SampleFrame instead.')
    if isfield(Exp, 'Mode'):
        raise ValueError('Exp.Mode is no longer supported. Use Exp.mwMode instead.')
    if isfield(Exp, 'lightBeam') and not isempty(Exp.lightBeam):
        raise ValueError('Photoselection (via Exp.lightBeam) is not supported.')
    if isfield(Exp, 'Ordering') and not isempty(Exp.Ordering):
        raise ValueError('Partial ordering (via Exp.Ordering) is not supported.')

    if not isfield(Exp, 'mwFreq') or isempty(Exp.mwFreq):
        if not isfield(Exp, 'Field'):
            raise ValueError('Please supply either the microwave frequency in Exp.mwFreq (for field sweeps) or the '
                             'magnetic field in Exp.Field (for frequency sweeps).')
        FieldSweep = False
    else:
        if isfield(Exp, 'Field'):
            if not isempty(Exp.Field):
                raise ValueError('Give either Exp.mwFreq (for a field sweep) or Exp.Frequency (for a frequency '
                                 'sweep), but not both.')
        FieldSweep = True

    if FieldSweep:
        if numel(Exp.mwFreq) != 1 or np.any(np.real(Exp.mwFreq) <= 0) or not _isreal(Exp.mwFreq):
            raise ValueError('Uninterpretable microwave frequency in Exp.mwFreq.')
        F.logmsg(1, '  field sweep, mw frequency %0.8g GHz', _scalar(Exp.mwFreq))
    else:
        if numel(Exp.Field) != 1 or np.any(np.real(Exp.Field) <= 0) or not _isreal(Exp.Field):
            raise ValueError('Uninterpretable magnetic field in Exp.Field.')
        F.logmsg(1, '  frequency sweep, magnetic field %0.8g mT', _scalar(Exp.Field))

    if not FieldSweep:
        Exp.mwPhase = -Exp.mwPhase

    SweepAutoRange = False
    if FieldSweep:
        if isfield(Exp, 'CenterSweep'):
            if isfield(Exp, 'Range'):
                F.logmsg(0, 'Using Exp.CenterSweep and ignoring Exp.Range.')
        else:
            if isfield(Exp, 'Range'):
                R = vec(np.asarray(Exp.Range, dtype=float))
                if R[0] >= R[1] or np.any(R < 0):
                    raise ValueError('Invalid sweep range!')
                Exp.CenterSweep = np.array([[np.mean(R), np.diff(R)[0]]])
            else:
                F.logmsg(1, '  automatic determination of sweep range')
                SweepAutoRange = True
        if not SweepAutoRange:
            CS = vec(np.asarray(Exp.CenterSweep, dtype=float))
            Exp.Range = CS[0] + np.array([[-1.0, 1.0]]) / 2 * CS[1]
    else:
        if isfield(Exp, 'mwCenterSweep'):
            if isfield(Exp, 'mwRange'):
                F.logmsg(0, 'Using Exp.mwCenterSweep and ignoring Exp.mwRange.')
        else:
            if isfield(Exp, 'mwRange'):
                R = vec(np.asarray(Exp.mwRange, dtype=float))
                if R[0] >= R[1] or np.any(R < 0):
                    raise ValueError('Invalid sweep range!')
                Exp.mwCenterSweep = np.array([[np.mean(R), np.diff(R)[0]]])
            else:
                F.logmsg(1, '  automatic determination of sweep range')
                SweepAutoRange = True
        if not SweepAutoRange:
            CS = vec(np.asarray(Exp.mwCenterSweep, dtype=float))
            Exp.mwRange = CS[0] + np.array([[-1.0, 1.0]]) / 2 * CS[1]

    if (not _isreal(Exp.nPoints)) or numel(Exp.nPoints) > 1 or _scalar(Exp.nPoints) < 2:
        raise ValueError('Problem with Exp.nPoints. Needs to be a number not smaller than 2.')
    nPoints = int(_scalar(Exp.nPoints))

    autoHarmonic = (not isfield(Exp, 'Harmonic')) or isempty(Exp.Harmonic) or _if(np.isnan(Exp.Harmonic))
    noBroadening = (not FastMotionRegime) and bool(np.all(lw == 0)) and bool(np.all(lwpp == 0))
    if autoHarmonic:
        if FieldSweep and not noBroadening:
            Exp.Harmonic = 0 if noBroadening else 1
        else:
            Exp.Harmonic = 0
    if not np.any(np.asarray(Exp.Harmonic) == np.array([-1, 0, 1, 2])):
        raise ValueError('Exp.Harmonic must be either 0, 1 or 2.')
    Harmonic = int(_scalar(Exp.Harmonic))
    if noBroadening and Harmonic != 0:
        raise ValueError('\n  No broadening given. Cannot compute spectrum with Exp.Harmonic=%d.\n' % Harmonic)

    if Exp.mwMode == 'perpendicular':
        ParallelMode = False
    elif Exp.mwMode == 'parallel':
        ParallelMode = True
    else:
        raise ValueError("Exp.mwMode must be either 'perpendicular' or 'parallel'.")
    F.logmsg(1, '  harmonic %d, %s mode', Harmonic, Exp.mwMode)

    Temperature = Exp.Temperature
    if not _if(np.isnan(Temperature)):
        if _if(np.isinf(Temperature)):
            raise ValueError('If given, Exp.Temperature must be a finite value.')
        if numel(Temperature) != 1:
            raise ValueError('garlic does not support spin polarization in Exp.Temperature.')

    ModAmp = np.asarray(Exp.ModAmp, dtype=float)
    if np.any(ModAmp < 0) or np.any(np.isnan(ModAmp)) or ModAmp.size != 1:
        raise ValueError('Exp.ModAmp must be either a single positive number or zero.')
    ModAmp = float(ModAmp.ravel()[0])
    if ModAmp > 0:
        if FieldSweep:
            F.logmsg(1, '  field modulation, amplitude %g mT', ModAmp)
            if Harmonic < 1:
                raise ValueError('With field modulation (Exp.ModAmp), Exp.Harmonic=0 does not work.')
            Exp.ModHarmonic = Exp.Harmonic
            Exp.ConvHarmonic = 0
            Exp.DerivHarmonic = 0
        else:
            raise ValueError('Exp.ModAmp cannot be used with frequency sweeps.')
    else:
        Exp.ModHarmonic = 0
        if ConvolutionBroadening:
            Exp.ConvHarmonic = Exp.Harmonic
            Exp.DerivHarmonic = 0
        else:
            Exp.ConvHarmonic = 0
            Exp.DerivHarmonic = Exp.Harmonic

    if isfield(Exp, 'CrystalSymmetry'):
        warnings.warn('Exp.CrystalSymmetry is not used by garlic.')
    if isfield(Exp, 'SampleFrame'):
        warnings.warn('Exp.SampleFrame is not used by garlic.')

    if isfield(Opt, 'Output'):
        raise ValueError('Options.Output is obsolete. Use Options.separate instead.')
    if not isfield(Opt, 'Stretch'):
        Opt.Stretch = 0.25
    if not isfield(Opt, 'MaxIterations'):
        Opt.MaxIterations = 15
    if not isfield(Opt, 'Accuracy'):
        Opt.Accuracy = 1e-12
    if isfield(Opt, 'PerturbOrder'):
        raise ValueError("Options.PerturbOrder is obsolete. Use Options.Method = 'perturb2' etc instead.")
    if not isfield(Opt, 'Method'):
        Opt.Method = 'exact'
    orders = {'exact': 0, 'perturb': 5, 'perturb1': 1, 'perturb2': 2, 'perturb3': 3, 'perturb4': 4, 'perturb5': 5}
    if not isinstance(Opt.Method, str) or Opt.Method not in orders:
        raise ValueError("Unknown method '%s'." % (Opt.Method,))
    PerturbOrder = orders[Opt.Method]
    if not isfield(Opt, 'AccumMethod') or isempty(Opt.AccumMethod):
        Opt.AccumMethod = 'explicit' if FastMotionRegime else 'binning'
    MaxIterations = int(_scalar(Opt.MaxIterations))
    Accuracy = _scalar(Opt.Accuracy)

    # Isotropic g
    gS = a2(Sys.g).astype(float)
    if _if(Sys.fullg):
        ev = np.linalg.eigvals(gS[0:3, :])
        giso = np.mean(ev)
        if np.iscomplexobj(giso) and giso.imag == 0:
            giso = giso.real
    else:
        giso = np.mean(gS[0, :])

    if FieldSweep:
        mwFreq = _scalar(Exp.mwFreq) * 1e9
        CentralResonance = (mwFreq * _c('planck')) / (_c('bmagn') * giso) * 1e3
    else:
        Field = _scalar(Exp.Field)
        CentralResonance = _c('bmagn') * giso * Field * 1e-3 / _c('planck')

    if FastMotionRegime:
        if FieldSweep:
            FastMotionLw = F.fastmotion(Sys, CentralResonance, Sys.tcorr, 'field')
        else:
            FastMotionLw = F.fastmotion(Sys, Exp.Field, Sys.tcorr, 'freq')
        if isinstance(FastMotionLw, tuple):
            FastMotionLw = FastMotionLw[0]
        FastMotionLw = np.asarray(FastMotionLw, dtype=float)
        if not FieldSweep:
            FastMotionLw = FastMotionLw / 1e3
        if np.all(FastMotionLw == 0):
            raise ValueError('All calculated fast-motion linewidths for rotational diffusion are zero. Please '
                             'provide full tensors, and not just isotropic averages.')
        elif np.any(FastMotionLw <= 0):
            raise ValueError('Some calculated fast-motion linewidths are negative. They must be positive.')

    nNuclei = int(Sys.nNuclei)
    if nNuclei > 0:
        I_all = vec(np.asarray(Sys.I, dtype=float))
        gn = vec(np.asarray(Sys.gn, dtype=float))
        A = a2(Sys.A).astype(float)
        if not _if(Sys.fullA):
            a_all = np.mean(A, axis=1) * 1e6
        else:
            a_all = np.zeros(nNuclei)
            for iNuc in range(nNuclei):
                Adiag_ = np.linalg.eigvals(A[iNuc * 3:iNuc * 3 + 3, :])
                a_all[iNuc] = np.real(np.mean(Adiag_)) * 1e6
        n_all = vec(np.asarray(Sys.n, dtype=float))
        if FieldSweep:
            zfSplitting = np.sum(np.abs(a_all) * (I_all + 0.5))
            if mwFreq < zfSplitting:
                print(('Microwave frequency (%f GHz) is smaller than\nlargest splitting '
                       'of levels at zero field (estimate %f GHz).\nSpectrum might be inaccurate.\n'
                       'Consider using the function pepper() instead.') % (mwFreq / 1e9, zfSplitting / 1e9))
        if np.any(I_all == 0):
            raise ValueError('Nuclei with spin 0 present. Cannot compute spectrum.')
        if np.any(a_all == 0):
            raise ValueError('Nuclei with hyperfine coupling 0 present. Cannot compute spectrum.')
        a_all = a_all * _c('planck')

    F.logmsg(1, 'Computing resonance shifts...')
    Shifts = [None] * nNuclei
    Amplitudes = [None] * nNuclei
    for iNucGrp in range(nNuclei):
        if n_all[iNucGrp] == 0:
            Shifts[iNucGrp] = np.array([0.0])
            Amplitudes[iNucGrp] = np.array([1.0])
            continue
        I_this, nn = F.equivcouple(I_all[iNucGrp], n_all[iNucGrp])
        I_this = vec(np.asarray(I_this, dtype=float))
        nn = vec(np.asarray(nn, dtype=float))
        nLines = (2 * I_all[iNucGrp] + 1) ** n_all[iNucGrp]
        nFSpins = nn.size
        aiso = a_all[iNucGrp]
        aiso = abs(aiso)
        Positions = np.zeros(0)
        Intensities = np.zeros(0)
        if FieldSweep:
            if PerturbOrder == 0:
                F.logmsg(1, '  Breit-Rabi solver, accuracy %g', Accuracy)
                maxIterationsDone = -1
                gammae = giso * _c('bmagn')
                h = _c('planck')
                for iF in range(nFSpins):
                    if nn[iF] == 0:
                        continue
                    I = I_this[iF]
                    mI = np.arange(-I, I + 0.5)
                    B_ = np.zeros(1)
                    RelChangeB = np.array([np.inf])
                    gamman = 0.0
                    for it in range(1, MaxIterations + 1):
                        nu = mwFreq + gamman / h * B_
                        q = 1 - (aiso / (2 * h * nu)) ** 2
                        arg = mI ** 2 + q * ((h * nu / aiso) ** 2 - (I + 1 / 2) ** 2)
                        if np.any(arg < 0):
                            raise ValueError('Hyperfine coupling too large compared to microwave frequency. Cannot '
                                             'compute resonance field positions. Use pepper() instead.')
                        Bnew = aiso / (gammae + gamman) / q * (-mI + np.sign(aiso) * np.sqrt(arg))
                        if _if(B_ != 0):
                            RelChangeB = 1 - Bnew / B_
                        B_ = Bnew
                        gamman = gn[iNucGrp] * _c('nmagn')
                        if _if(np.abs(RelChangeB) < Accuracy):
                            break
                    if it > maxIterationsDone:
                        maxIterationsDone = it
                    if it == MaxIterations:
                        raise ValueError("Breit-Rabi solver didn't converge after %d iterations!" % it)
                    Positions = np.concatenate([Positions, B_])
                    Intensities = np.concatenate([Intensities, nn[iF] / nLines * np.ones(B_.size)])
                F.logmsg(1, '  maximum %d iterations done', maxIterationsDone)
            else:
                F.logmsg(1, '  perturbation expansion, order %d, accuracy %g', PerturbOrder, Accuracy)
                maxIterationsDone = -1
                pre = aiso * (aiso / (giso * _c('bmagn') + gn[iNucGrp] * _c('nmagn')) / 2) ** np.arange(1, 5)
                for iF in range(nFSpins):
                    if nn[iF] == 0:
                        continue
                    I = I_this[iF]
                    for mI in np.arange(-I, I + 0.5):
                        c = np.zeros(6)
                        c[5] = pre[3] * mI * (1 + 6 * (I - 1) * I * (I + 1) * (I + 2) - 10 * (-3 + 2 * I * (I + 1)) * mI ** 2
                                              + 14 * mI ** 4)
                        c[4] = pre[2] * (I - 2 * I ** 3 - I ** 4 + 6 * (I ** 2 + I - 1) * mI ** 2 - 5 * mI ** 4)
                        c[3] = pre[1] * mI * (1 - 2 * I * (I + 1) + 2 * mI ** 2)
                        c[2] = pre[0] * (I * (I + 1) - mI ** 2)
                        c[1] = aiso * mI - mwFreq * _c('planck')
                        c[0] = giso * _c('bmagn')
                        n = PerturbOrder + 1
                        c = c[:n]
                        dc = np.arange(n - 1, 0, -1) * c[:n - 1]
                        Bb = -c[1] / c[0]
                        dB = Bb
                        for it in range(1, MaxIterations + 1):
                            if abs(dB / Bb) < Accuracy:
                                break
                            dB = np.polyval(c, Bb) / np.polyval(dc, Bb)
                            Bb = Bb - dB
                        if it > maxIterationsDone:
                            maxIterationsDone = it
                        if it == MaxIterations:
                            raise ValueError('Newton-Raphson has convergence problem!')
                        Positions = np.append(Positions, Bb)
                        Intensities = np.append(Intensities, nn[iF] / nLines)
                F.logmsg(1, '  maximum %d iterations done', maxIterationsDone)
            Positions = Positions * 1e3
        else:
            if PerturbOrder == 0:
                F.logmsg(1, '  Breit-Rabi formula')
                B_ = Field * 1e-3
                gebB = giso * _c('bmagn') * B_
                gnbB = gn[iNucGrp] * _c('nmagn') * B_
                for iF in range(nFSpins):
                    if nn[iF] == 0:
                        continue
                    I = I_this[iF]
                    mI = np.arange(-I, I + 0.5)
                    alpha = (gebB + gnbB) / aiso / (I + 1 / 2)
                    E1 = -aiso / 4 - gnbB * (mI + 1 / 2) + (I + 1 / 2) * aiso / 2 * np.sqrt(
                        1 + 2 * (mI + 1 / 2) / (I + 1 / 2) * alpha + alpha ** 2)
                    E2 = -aiso / 4 - gnbB * (mI - 1 / 2) - (I + 1 / 2) * aiso / 2 * np.sqrt(
                        1 + 2 * (mI - 1 / 2) / (I + 1 / 2) * alpha + alpha ** 2)
                    nu_ = (E1 - E2) / _c('planck')
                    Positions = np.concatenate([Positions, nu_])
                    Intensities = np.concatenate([Intensities, nn[iF] / nLines * np.ones(nu_.size)])
            else:
                F.logmsg(1, '  perturbation theory, order %d', PerturbOrder)
                B_ = Field * 1e-3
                pre = aiso * (aiso / (giso * _c('bmagn') * B_ + gn[iNucGrp] * _c('nmagn') * B_) / 2) ** np.arange(1, 5)
                for iF in range(nFSpins):
                    if nn[iF] == 0:
                        continue
                    I = I_this[iF]
                    for mI in np.arange(-I, I + 0.5):
                        c = np.zeros(6)
                        c[0] = giso * _c('bmagn') * B_
                        c[1] = aiso * mI
                        c[2] = pre[0] * (I * (I + 1) - mI ** 2)
                        c[3] = pre[1] * mI * (1 - 2 * I * (I + 1) + 2 * mI ** 2)
                        c[4] = pre[2] * (I - 2 * I ** 3 - I ** 4 + 6 * (I ** 2 + I - 1) * mI ** 2 - 5 * mI ** 4)
                        c[5] = pre[3] * mI * (1 + 6 * (I - 1) * I * (I + 1) * (I + 2) - 10 * (-3 + 2 * I * (I + 1)) * mI ** 2
                                              + 14 * mI ** 4)
                        dE = np.sum(c[:PerturbOrder + 1])
                        nu_ = dE / _c('planck')
                        Positions = np.append(Positions, nu_)
                        Intensities = np.append(Intensities, nn[iF] / nLines)
        Shifts[iNucGrp] = Positions - CentralResonance
        Amplitudes[iNucGrp] = Intensities
        F.logmsg(1, '  spin group %d: %d F spins, %d lines', iNucGrp + 1, nFSpins, Shifts[iNucGrp].size)

    nPeaks = 1
    posmax = CentralResonance
    posmin = CentralResonance
    for s in Shifts:
        nPeaks = nPeaks * s.size
        posmax = posmax + np.max(s)
        posmin = posmin + np.min(s)
    F.logmsg(1, '  total %d lines', nPeaks)

    if FastMotionRegime:
        LorentzianLw = FastMotionLw + lw[1]
        lw[1] = 0
        maxLw = max(np.max(FastMotionLw), lw[0])
    else:
        LorentzianLw = lw[1]
        maxLw = np.max(lw)
    Sys.lw = lw.reshape(1, -1)

    if SweepAutoRange:
        if FieldSweep:
            posrange = (posmax - posmin) * Opt.Stretch
            minrange = 1
            Rng = np.array([posmin, posmax]) + np.array([-1, 1]) * max(5 * maxLw, posrange, minrange)
            Rng[0] = max(Rng[0], 0)
            Exp.Range = Rng.reshape(1, -1)
            F.logmsg(1, '  automatic field range from %g mT to %g mT', Rng[0], Rng[1])
        else:
            posrange = (posmax - posmin) * Opt.Stretch
            minrange = 1e6
            Rng = np.array([posmin, posmax]) + np.array([-1, 1]) * max(5 * maxLw / 1e3, posrange, minrange)
            Rng[0] = max(Rng[0], 0)
            Rng = Rng / 1e9
            Exp.mwRange = Rng.reshape(1, -1)
            F.logmsg(1, '  automatic frequency range from %g GHz to %g GHz', Rng[0], Rng[1])

    if FieldSweep:
        F.logmsg(1, '  spectral spread %g mT\n  g=%g resonance at %g mT', posmax - posmin, giso, CentralResonance)
    else:
        F.logmsg(1, '  spectral spread %g MHz\n  g=%g resonance at %g GHz', (posmax - posmin) / 1e6, giso,
                 CentralResonance / 1e9)

    F.logmsg(1, 'Combining line shifts and line multiplicities...')
    if nPeaks > 1:
        Positions = np.asarray(F.allcombinations(*Shifts, '+')) + CentralResonance
        Intensities = np.asarray(F.allcombinations(*Amplitudes, '*'))
    else:
        Positions = CentralResonance
        Intensities = 1.0
    Positions = np.asarray(Positions, dtype=float)
    Intensities = np.asarray(Intensities, dtype=float)
    if not FieldSweep:
        Positions = Positions / 1e9

    F.logmsg(1, 'Computing overall line intensities...')
    if ParallelMode:
        Intensities = np.zeros(Intensities.shape)
    else:
        g1mean2 = giso ** 2
        TransitionRate = (8 * np.pi ** 2) * g1mean2 * (_c('bmagn') / _c('planck') / 1e9 / 2) ** 2
        Intensities = Intensities * TransitionRate
        if FieldSweep:
            dBdE = _c('planck') / (giso * _c('bmagn')) * 1e9
            Intensities = Intensities * dBdE
        if np.all(np.isfinite(Exp.Temperature)):
            T = _scalar(Exp.Temperature)
            if FieldSweep:
                DeltaE = _c('planck') * _scalar(Exp.mwFreq) * 1e9
            else:
                DeltaE = _c('bmagn') * giso * Field * 1e-3
            e = np.exp(-DeltaE / _c('boltzm') / T)
            Population = np.array([1, e]) / (1 + e)
            Polarization = Population[0] - Population[1]
            Intensities = Intensities * Polarization

    SweepRange = vec(np.asarray(Exp.Range if FieldSweep else Exp.mwRange, dtype=float))
    xAxis = _linspace(SweepRange[0], SweepRange[1], nPoints)
    ConvHarmonic = int(_scalar(Exp.ConvHarmonic))
    mwPhase = _scalar(Exp.mwPhase)
    LorentzianLw = np.asarray(LorentzianLw, dtype=float)

    if Opt.AccumMethod == 'template':
        F.logmsg(1, 'Constructing spectrum using Lorentzian lineshape template...')
        if _if(LorentzianLw == 0):
            raise ValueError('Cannot use templated linshape accumulation with zero Lorentzian linewidth.')
        if mwPhase != 0:
            raise ValueError('Cannot use templated lineshape accumulation with non-zero Exp.mwPhase.')
        dxFine = min(xAxis[1] - xAxis[0], np.min(LorentzianLw) / 5)
        nPointsFine = int(_mround((SweepRange[1] - SweepRange[0]) / dxFine + 1))
        xAxisFine = _linspace(SweepRange[0], SweepRange[1], nPointsFine)
        dxFine = xAxisFine[1] - xAxisFine[0]
        x0T = 1e5
        wT = x0T / 20
        xT = np.arange(0, 2 * x0T).reshape(1, -1)
        Template = F.lorentzian(xT, x0T, wT, Harmonic - 1)
        if LorentzianLw.size == 1:
            LorentzianLw = LorentzianLw.ravel()[0] * np.ones(Positions.shape)
        spec = F.lisum1i(Template, x0T, wT, Positions, Intensities, LorentzianLw, xAxisFine)
        Exp.ConvHarmonic = 0
        ConvHarmonic = 0
        LorentzianLw = np.asarray(0.0)
    elif Opt.AccumMethod == 'explicit':
        F.logmsg(1, 'Constructing spectrum with explicit Lorentzian lineshapes...')
        if _if(LorentzianLw == 0):
            raise ValueError('Cannot use axplicit accumulation with zero Lorentzian linewidth.')
        dxFine = min(xAxis[1] - xAxis[0], np.min(LorentzianLw) / 5)
        nPointsFine = int(_mround((SweepRange[1] - SweepRange[0]) / dxFine + 1))
        xAxisFine = _linspace(SweepRange[0], SweepRange[1], nPointsFine)
        dxFine = xAxisFine[1] - xAxisFine[0]
        spec = 0
        Pv, Iv = vec(Positions), vec(Intensities)
        if LorentzianLw.size == 1:
            lwL = float(LorentzianLw.ravel()[0])
            for iLine in range(Pv.size):
                spec = spec + Iv[iLine] * F.lorentzian(xAxisFine, Pv[iLine], lwL, ConvHarmonic, mwPhase)
        else:
            Lv = vec(LorentzianLw)
            for iLine in range(Pv.size):
                spec = spec + Iv[iLine] * F.lorentzian(xAxisFine, Pv[iLine], Lv[iLine], ConvHarmonic, mwPhase)
        spec = np.asarray(spec).reshape(1, -1)
        Exp.mwPhase = 0
        mwPhase = 0.0
        Exp.ConvHarmonic = 0
        ConvHarmonic = 0
        LorentzianLw = np.asarray(0.0)
    elif Opt.AccumMethod == 'binning':
        F.logmsg(1, 'Constructing stick spectrum using binning...')
        if FastMotionRegime:
            raise ValueError("Cannot use delta binning (Options.AccumMethod='binning' in the fast-motion regime.")
        expandFactor = 1
        nPointsFine = (nPoints - 1) * expandFactor + 1
        xAxisFine = _linspace(SweepRange[0], SweepRange[1], nPointsFine)
        dxFine = xAxisFine[1] - xAxisFine[0]
        spec = _constructstickspectrum(Positions, Intensities, SweepRange, nPointsFine)
        spec = spec / dxFine
    else:
        raise ValueError("\n  Unknown method '%s' in Options.AccumMethod.\n" % (Opt.AccumMethod,))

    if ConvolutionBroadening:
        fwhmL = LorentzianLw
        fwhmG = lw[0]
        if _if(fwhmL > 0):
            HarmonicL = ConvHarmonic
            mwPhaseL = mwPhase
            HarmonicG = 0
            mwPhaseG = 0
        else:
            HarmonicL = 0
            mwPhaseL = 0
            HarmonicG = ConvHarmonic
            mwPhaseG = mwPhase
        if FieldSweep:
            unitstr = 'mT'
        else:
            fwhmL = fwhmL / 1e3
            fwhmG = fwhmG / 1e3
            unitstr = 'GHz'
        if _if(fwhmL != 0):
            fL = float(np.ravel(fwhmL)[0]) if np.size(fwhmL) == 1 else fwhmL
            if _if(fwhmL > 2 * dxFine):
                F.logmsg(1, 'Convoluting with Lorentzian (FWHM %g %s, derivative %d)...', fL, unitstr, HarmonicL)
                spec = F.convspec(spec, dxFine, fL, HarmonicL, 0, mwPhaseL)
            else:
                if HarmonicL != 0:
                    raise ValueError(('Lorentzian linewidth (FWHM %g %s) is smaller than 2 increments (2x%g = %g %s) - '
                                      'cannot do convolution.\nIncrease linewidth or decrease increment.')
                                     % (fL, unitstr, dxFine, 2 * dxFine, unitstr))
        if fwhmG != 0:
            if fwhmG > 2 * dxFine:
                F.logmsg(1, 'Convoluting with Gaussian (FWHM %g %s, derivative %d)...', fwhmG, unitstr, HarmonicG)
                spec = F.convspec(spec, dxFine, fwhmG, HarmonicG, 1, mwPhaseG)
            else:
                if HarmonicG != 0:
                    raise ValueError(('Gaussian linewidth (FWHM %g %s) is smaller than 2 increments (2x%g = %g %s) - '
                                      'cannot do convolution.\nIncrease linewidth or decrease increment.')
                                     % (fwhmG, unitstr, dxFine, 2 * dxFine, unitstr))

    if xAxisFine.size != xAxis.size:
        F.logmsg(1, 'Re-interpolation (%d -> %d points)...', xAxisFine.size, xAxis.size)
        spec = np.interp(xAxis, xAxisFine, vec(spec)).reshape(1, -1)

    if FieldSweep:
        if ModAmp > 0:
            F.logmsg(1, 'Applying field modulation (%g mT amplitude)...', ModAmp)
            spec = F.fieldmod(xAxis, spec, ModAmp, Exp.ModHarmonic)

    xAxis = xAxis.reshape(1, -1)
    if nargout == 1:
        out = spec
    elif nargout == 2:
        out = (xAxis, spec)
    elif nargout == 3:
        info = Struct()
        info.resfields = Positions
        out = (xAxis, spec, info)
    else:
        out = None
    if EasySpinLogLevel is not None and EasySpinLogLevel >= 1:
        import datetime
        F.logmsg(1, '=end=garlic=======%s=================\n', datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S'))
    config.LogLevel = None
    return out
