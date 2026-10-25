"""Simulate cw EPR spectra of powders and crystals in the rigid limit."""
import numpy as np

from .._registry import F, nout
from .._struct import Struct
from .._mat import a2, vec, isempty, numel
from .._matlab import linspace
from ..config import config


def _t(x):
    a = np.asarray(x)
    return a.size > 0 and bool(np.all(a != 0))


def _num(x):
    return float(np.asarray(x, dtype=float).ravel()[0])


def _opt(Opt, name, choices):
    """parseoption -> 1-based index, raising on an invalid value."""
    r = F.parseoption(Opt, name, choices, nargout=2)
    idx, err = (r[0], r[1]) if isinstance(r, tuple) else (r, '')
    if err is not None and not isempty(err):
        raise ValueError(err)
    return int(_num(idx))


@nout(-1)
def pepper(Sys=None, Exp=None, Opt=None, nargout=2, plot=False):
    """nargout 1: spec; 2: (xAxis, spec); 3: (xAxis, spec, info). plot=True draws the spectrum
    (MATLAB plots when called without outputs)."""
    if Sys is None:
        raise ValueError('Please supply a spin system as first input argument.')
    if Exp is None:
        raise ValueError('Please supply experimental parameters as second input argument.')
    if nargout > 3:
        raise ValueError('Too many output arguments.')
    if Opt is None or isempty(Opt):
        Opt = Struct()
    if not isinstance(Sys, (dict, list, tuple)):
        raise ValueError('The first input (Sys) must be a structure or a cell array of structures.')
    if not isinstance(Exp, dict):
        raise ValueError('The second input (Exp) must be a structure.')
    if not isinstance(Opt, dict):
        raise ValueError('The third input (Opt) must be a structure.')
    Exp, Opt = Struct(Exp), Struct(Opt)
    if 'Verbosity' not in Opt:
        Opt.Verbosity = 0.0
    config.LogLevel = Opt.Verbosity
    Exp.FrequencySweep = 'mwFreq' not in Exp and 'Field' in Exp
    if Exp.FrequencySweep:
        SweepAutoRange = (isempty(Exp.get('mwRange', [])) and isempty(Exp.get('mwCenterSweep', [])))
    else:
        SweepAutoRange = (isempty(Exp.get('Range', [])) and isempty(Exp.get('CenterSweep', [])))
    if 'IsoCutoff' not in Opt:
        Opt.IsoCutoff = 1e-3
    if 'separate' not in Opt:
        Opt.separate = ''
    separateOutput = _opt(Opt, 'separate', ['', 'components', 'transitions', 'orientations', 'sites'])
    summedOutput = separateOutput == 1
    separateTransitionSpectra = separateOutput == 3
    separateOrientationSpectra = separateOutput == 4
    separateSiteSpectra = separateOutput == 5
    separateSpectra = not summedOutput

    singleIsotopologue = isinstance(Sys, dict) and 'singleiso' in Sys and _t(Sys.singleiso)
    if not singleIsotopologue:
        thirdOutput = nargout >= 3
        xAxis, spec, info = F.compisoloop(F.pepper, Sys, Exp, Opt, SweepAutoRange, thirdOutput, separateSpectra)[:3]
        if plot or nargout == 0:
            F.cwepr_plot(xAxis, spec, Exp)
            if nargout == 0:
                return None
        if nargout == 1:
            return spec
        if nargout == 2:
            return xAxis, spec
        return xAxis, spec, info

    F.logmsg(1, '=begin=pepper====================================')
    F.logmsg(2, '  log level %d', _num(Opt.Verbosity))
    F.logmsg(1, '-general-----------------------------------------------')
    Sys, err = F.validatespinsys(Sys)
    if err is not None and not isempty(err):
        raise ValueError(err)
    Sys = Struct(Sys)
    if 'n' in Sys and np.any(np.asarray(Sys.n, dtype=float) > 1):
        raise ValueError('pepper does not support sets of equivalent nuclei (Sys.n>1).')
    StrainWidths = any(np.any(np.asarray(Sys[f], dtype=float) > 0) for f in ('HStrain', 'DStrain', 'gStrain', 'AStrain'))
    lw = vec(np.asarray(Sys.lw, dtype=float)).copy()
    ConvolutionBroadening = bool(np.any(lw > 0))
    F.logmsg(1, '  system with %d spin(s) and %d states', numel(F.spinvec(Sys)), _num(F.hsdim(Sys)))
    if StrainWidths:
        F.logmsg(1, '  strain widths given')
    DefaultExp = Struct(CenterSweep=np.nan, Range=np.nan, mwCenterSweep=np.nan, mwRange=np.nan, nPoints=1024.0,
                        Temperature=np.nan, Harmonic=np.nan, mwMode='perpendicular', ModAmp=0.0, mwPhase=0.0,
                        lightBeam='', Ordering=np.zeros((0, 0)), SampleRotation=np.zeros((0, 0)),
                        SampleFrame=np.zeros((0, 0)), CrystalSymmetry='', MolFrame=np.zeros((0, 0)))
    Exp = Struct(F.adddefaults(Exp, DefaultExp))
    if 'Orientations' in Exp:
        raise ValueError('Exp.Orientations is no longer supported, use Exp.SampleFrame instead.')
    if 'CrystalOrientation' in Exp:
        raise ValueError('Exp.CrystalOrientation is no longer supported, use Exp.SampleFrame/Exp.MolFrame instead.')
    if 'Mode' in Exp:
        raise ValueError('Exp.Mode is no longer supported. Use Exp.mwMode instead.')
    if 'mwFreq' not in Exp or isempty(Exp.mwFreq):
        if 'Field' not in Exp:
            raise ValueError('Please supply either the microwave frequency in Exp.mwFreq (for field sweeps) or the '
                             'magnetic field in Exp.Field (for frequency sweeps).')
        FieldSweep = False
    else:
        if 'Field' in Exp and not isempty(Exp.Field):
            raise ValueError('Give either Exp.mwFreq (for a field sweep) or Exp.Field (for a frequency sweep), but not both.')
        FieldSweep = True
    if not FieldSweep:
        lw = lw / 1e3
    Sys.lw = lw
    if FieldSweep:
        mw = np.asarray(Exp.mwFreq)
        if mw.size != 1 or np.any(mw <= 0) or np.iscomplexobj(mw):
            raise ValueError('Uninterpretable microwave frequency in Exp.mwFreq.')
        F.logmsg(1, '  field sweep, mw frequency %0.8g GHz', _num(Exp.mwFreq))
    else:
        fl = np.asarray(Exp.Field)
        if fl.size != 1 or np.iscomplexobj(fl):
            raise ValueError('Uninterpretable magnetic field in Exp.Field.')
        F.logmsg(1, '  frequency sweep, magnetic field %0.8g mT', _num(Exp.Field))

    if FieldSweep:
        if np.all(np.isnan(np.asarray(Exp.CenterSweep, dtype=float))) and np.all(np.isnan(np.asarray(Exp.Range, dtype=float))):
            S = vec(np.asarray(Sys.S, dtype=float))
            L = np.asarray(Sys.get('L', np.zeros(0)), dtype=float)
            if S.size == 1 and S[0] == 0.5 and not np.any(L != 0):
                F.logmsg(1, '  automatic determination of sweep range')
                I = vec(np.asarray(F.nucspin(Sys.Nucs), dtype=float)) if 'Nucs' in Sys and not isempty(Sys.Nucs) else np.zeros(0)
                if I.size:
                    Aabs = np.abs(a2(np.asarray(Sys.A, dtype=float)))
                    Amax = np.max(Aabs, axis=1)
                    if _t(Sys.fullA):
                        Amax = np.max(Amax.reshape((3, -1), order='F'), axis=0)
                else:
                    Amax = np.zeros(1)
                hf = float(np.sum(I * Amax)) * 1e6 if I.size else 0.0
                if _t(Sys.fullg):
                    gm = a2(np.asarray(Sys.g, dtype=float))
                    g = np.concatenate([np.linalg.eigvals(gm[3 * k:3 * k + 3, :]).real for k in range(int(Sys.nElectrons))])
                else:
                    g = vec(np.asarray(Sys.g, dtype=float))
                gmax, gmin = np.max(g), np.min(g)
                planck, bmagn = _num(F.planck()), _num(F.bmagn())
                minB = planck * (_num(Exp.mwFreq) * 1e9 - hf) / bmagn / gmax / 1e-3
                maxB = planck * (_num(Exp.mwFreq) * 1e9 + hf) / bmagn / gmin / 1e-3
                Center = (maxB + minB) / 2
                Sweep = maxB - minB + 3 * np.max(lw)
                if Sweep == 0:
                    Sweep = 5 * np.max(lw)
                if Sweep == 0:
                    Sweep = 10.0
                Exp.CenterSweep = np.array([[Center, 1.25 * Sweep]])
            else:
                raise ValueError('Cannot automatically determine field range.\nPlease provide either Exp.CenterSweep or Exp.Range.')
        cs = vec(np.asarray(Exp.CenterSweep, dtype=float))
        if np.all(~np.isnan(cs)):
            Exp.Range = cs[0] + np.array([[-1.0, 1.0]]) * cs[1] / 2
            if vec(Exp.Range)[0] < 0:
                raise ValueError('Lower field limit from Exp.CenterSweep cannot be negative.')
        Rg = vec(np.asarray(Exp.Range))
        if np.all(~np.isnan(Rg)):
            if np.any(np.diff(Rg) <= 0) or np.any(~np.isfinite(Rg)) or np.iscomplexobj(Rg):
                raise ValueError('Exp.Range is not valid!')
            if np.any(Rg < 0):
                raise ValueError('Negative magnetic fields in Exp.Range are not possible.')
    else:
        cs = vec(np.asarray(Exp.mwCenterSweep, dtype=float))
        if np.all(~np.isnan(cs)):
            Exp.mwRange = np.maximum(cs[0] + np.array([[-1.0, 1.0]]) * cs[1] / 2, 0)
        Rg = vec(np.asarray(Exp.mwRange))
        if np.all(~np.isnan(Rg)):
            if np.diff(Rg)[0] <= 0 or np.any(~np.isfinite(Rg)) or np.iscomplexobj(Rg):
                raise ValueError('Exp.mwRange is not valid!')
            if np.any(Rg < 0):
                raise ValueError('Exp.mwRange cannot be negative.')
    nP = np.asarray(Exp.nPoints)
    if np.iscomplexobj(nP) or nP.size > 1 or float(nP.ravel()[0]) < 2:
        raise ValueError('Problem with Exp.nPoints. Needs to be a number not smaller than 2.')
    nPoints = int(_num(Exp.nPoints))
    if FieldSweep:
        Rg = vec(np.asarray(Exp.Range, dtype=float))
        F.logmsg(1, '  frequency %g GHz, field range [%g %g] mT, %d points', _num(Exp.mwFreq), Rg[0], Rg[1], nPoints)
    elif not SweepAutoRange:
        Rg = vec(np.asarray(Exp.mwRange, dtype=float))
        F.logmsg(1, '  field %g mT, frequency range [%g %g] GHz, %d points', _num(Exp.Field), Rg[0], Rg[1], nPoints)
    else:
        F.logmsg(1, '  field %g mT, automatic frequency range, %d points', _num(Exp.Field), nPoints)

    autoHarmonic = 'Harmonic' not in Exp or isempty(Exp.Harmonic) or np.isnan(_num(Exp.Harmonic))
    noBroadening = not StrainWidths and not ConvolutionBroadening
    if autoHarmonic:
        Exp.Harmonic = 1.0 if (FieldSweep and not noBroadening) else 0.0
    Harmonic = _num(Exp.Harmonic)
    if Harmonic not in (-1.0, 0.0, 1.0, 2.0):
        raise ValueError('Exp.Harmonic must be either 0, 1 or 2.')
    if noBroadening and Harmonic != 0:
        raise ValueError(f'\n  No broadening given. Cannot compute spectrum with Exp.Harmonic={int(Harmonic)}.\n')
    ma = np.asarray(Exp.ModAmp, dtype=float)
    if np.any(ma < 0) or np.any(np.isnan(ma)) or ma.size != 1:
        raise ValueError('Exp.ModAmp must be either a single positive number or zero.')
    ModAmp = float(ma.ravel()[0])
    if ModAmp > 0:
        if not FieldSweep:
            raise ValueError('Exp.ModAmp cannot be used with frequency sweeps.')
        if Harmonic < 1:
            raise ValueError('With field modulation (Exp.ModAmp), Exp.Harmonic=0 does not work.')
        ModHarmonic, ConvHarmonic, DerivHarmonic = Harmonic, 0.0, 0.0
        F.logmsg(1, '  explicit field modulation, amplitude %g mT, harmonic %d', ModAmp, Harmonic)
    else:
        ModHarmonic = 0.0
        if ConvolutionBroadening:
            ConvHarmonic, DerivHarmonic = Harmonic, 0.0
        else:
            ConvHarmonic, DerivHarmonic = 0.0, Harmonic
        F.logmsg(1, '  no explicit field modulation, derivative %d', Harmonic)
    Exp.ModHarmonic, Exp.ConvHarmonic, Exp.DerivHarmonic = ModHarmonic, ConvHarmonic, DerivHarmonic   # as MATLAB's Exp
    mwPhase = _num(Exp.mwPhase)
    if not FieldSweep:
        mwPhase = -mwPhase
        Exp.mwPhase = mwPhase
    if isinstance(Exp.mwMode, str) and Exp.mwMode:
        if Exp.mwMode not in ('perpendicular', 'parallel'):
            raise ValueError("Exp.mwMode must be either 'perpendicular' or 'parallel'.")
        F.logmsg(1, '  resonator mode: %s', Exp.mwMode)
    else:
        F.logmsg(1, '  resonator mode: custom')
    nonEquiPops = 'initState' in Sys and not isempty(Sys.initState)
    if nonEquiPops:
        msg = '  user-specified non-equilibrium state'
    else:
        if numel(Exp.Temperature) != 1:
            raise ValueError('If given, Exp.Temperature must be a single number.')
        T = _num(Exp.Temperature)
        msg = '  temperature %g K' % T if np.isfinite(T) else '  no temperature'
    F.logmsg(1, msg)

    Exp, Opt = (Struct(x) for x in F.p_sampletype(Exp, Opt)[:2])
    crystalSample = _t(Opt.crystalSample)
    for o in ('Convolution', 'Width'):
        if o in Opt:
            raise ValueError(f'Options.{o} is obsolete. Please remove from code!')
    if 'nKnots' in Opt:
        raise ValueError('Options.nKnots is obsolete. Use Options.GridSize instead, e.g. Options.GridSize = 91.')
    if 'Symmetry' in Opt:
        raise ValueError("Options.Symmetry is obsolete. Use Options.GridSymmetry instead, e.g. Options.GridSymmetry = 'D2h'.")
    if 'nSpline' in Opt:
        raise ValueError('Options.nSpline is obsolete. Use a second number in Options.GridSize instead, e.g. '
                         'Options.GridSize = [19 5] for Options.nSpline = 5.')
    if 'Perturb' in Opt:
        raise ValueError("Options.Perturb is obsolete. Use Opt.Method='perturb' or Opt.Method='hybrid' instead.")
    if 'Output' in Opt:
        raise ValueError('Options.Output is obsolete. Use Options.separate instead.')
    nR = len(Opt.R_L2S) if isinstance(Opt.R_L2S, list) else 1
    DefaultOpt = Struct(Verbosity=0.0, GridSymmetry='', GridFrame=np.zeros((0, 0)), Method='matrix',
                        separate='orientations' if (crystalSample and nR > 1) else '', Smoothing=2.0,
                        GridSizeMinimum=10.0, Debug=0.0, Intensity='on', BruteForce=0.0, ImmediateBinning=0.0,
                        PaddingMultiplier=3.0, GridSize=np.array([[19.0, 4.0]]))
    Opt = Struct(F.adddefaults(Opt, DefaultOpt))
    if FieldSweep:
        Method = _opt(Opt, 'Method', ['eig', 'matrix', 'perturb', 'perturb1', 'perturb2', 'hybrid'])
    else:
        Method = _opt(Opt, 'Method', ['matrix', 'perturb', 'perturb1', 'perturb2', 'hybrid']) + 10
    useEigenFields = Method == 1
    usePerturbationTheory = Method in (3, 4, 5, 12, 13, 14)
    ImmediateBinning = _t(Opt.ImmediateBinning)
    if ImmediateBinning and not usePerturbationTheory:
        raise ValueError('Opt.ImmediateBinning works only with perturbation theory.')
    if usePerturbationTheory and nonEquiPops:
        raise ValueError('Perturbation theory not available for systems with non-equilibrium populations.')
    GridSize = vec(np.asarray(Opt.GridSize, dtype=float)).copy()
    if GridSize[0] < _num(Opt.GridSizeMinimum):
        raise ValueError(f'Options.GridSize must not be less than {int(_num(Opt.GridSizeMinimum))}.')
    if GridSize.size < 2:
        GridSize = np.array([GridSize[0], 4.0])
    Opt.GridSize = GridSize.reshape(1, -1)
    if not crystalSample:
        if separateOrientationSpectra:
            raise ValueError("\nCannot return separate orientations for non-crystal spectra (Opt.separate='orientations')."
                             "\nUse other setting for Opt.separate.\n")
        if separateSiteSpectra:
            raise ValueError("\nCannot return separate sites for non-crystal spectra (Opt.separate='sites').\n"
                             "Use other setting for Opt.separate.\n")
    elif separateTransitionSpectra:
        raise ValueError("\nCannot return separate transitions for crystal spectra (Opt.separate='transitions').\n"
                         "  Use other setting for Opt.separate.\n")
    # EasySpin tests ~summedSpectra here, an undefined variable (it errors); summedOutput is meant
    if ImmediateBinning and not summedOutput:
        raise ValueError("When using Opt.ImmediateBinning, only Opt.separate='' is possible")
    anisotropicIntensities = _opt(Opt, 'Intensity', ['off', 'on']) - 1
    Opt.Intensity = float(anisotropicIntensities)
    Exp, Opt, nOrientations = F.p_gridsetup(Sys, Exp, Opt)[:3]
    Exp, Opt, nOrientations = Struct(Exp), Struct(Opt), int(_num(nOrientations))

    # ---------------------------------------------------------------- resonances
    F.logmsg(1, '-resonances--------------------------------------------')
    Range = vec(np.asarray(Exp.Range, dtype=float)) if FieldSweep else None
    spec = None
    if FieldSweep:
        if Method == 1:
            anisotropicWidths = False
            if StrainWidths:
                F.logmsg(-np.inf, 'WARNING: Options.Method: eigenfields method -> strains are ignored!')
            Exp1 = Struct(Exp)
            Exp1.Range = np.array([[0.0, 1e8]])
            Pdat, Idat = F.resfields_eig(Sys, Exp1, Opt, nargout=2)[:2]
            Wdat = np.zeros((0, 0))
            Transitions = np.zeros((0, 0))
            if nOrientations == 1:
                Pdat, Idat = [Pdat], [Idat]
            nReson = sum(np.size(p) for p in Pdat)
            F.logmsg(1, '  %d resonance in total (%g per orientation)', nReson, nReson / nOrientations)
        else:
            Exp1 = Struct(Exp)
            sr = Range + 0.2 * np.diff(Range)[0] * np.array([-1.0, 1.0])
            sr[sr < 0] = 0
            Exp1.SearchRange = sr.reshape(1, -1)
            Exp1.AccumWeights = Exp.OriWeights
            if Method in (2, 6):
                Pdat, Idat, Wdat, Transitions = F.resfields(Sys, Exp1, Opt, nargout=4)[:4]
            elif Method in (3, 5):
                Opt.PerturbOrder = 2.0
                Pdat, Idat, Wdat, Transitions, spec = F.resfields_perturb(Sys, Exp1, Opt)[:5]
            else:
                Opt.PerturbOrder = 1.0
                Pdat, Idat, Wdat, Transitions, spec = F.resfields_perturb(Sys, Exp1, Opt)[:5]
            anisotropicWidths = (not isempty(Wdat)) and np.nanmax(np.asarray(Wdat, dtype=float)) > 0
    else:
        Exp1 = Struct(Exp)
        Exp1.AccumWeights = Exp.OriWeights
        if Method in (11, 15):
            Pdat, Idat, Wdat, Transitions = F.resfreqs_matrix(Sys, Exp1, Opt, nargout=4)[:4]
        elif Method in (12, 14):
            Opt.PerturbOrder = 2.0
            Pdat, Idat, Wdat, Transitions, spec = F.resfreqs_perturb(Sys, Exp1, Opt)[:5]
        else:
            Opt.PerturbOrder = 1.0
            Pdat, Idat, Wdat, Transitions, spec = F.resfreqs_perturb(Sys, Exp1, Opt)[:5]
        Pdat = np.asarray(Pdat) / 1e3
        Wdat = np.asarray(Wdat) / 1e3
        anisotropicWidths = (not isempty(Wdat)) and np.nanmax(np.asarray(Wdat, dtype=float)) > 0
    if not useEigenFields:
        Pdat = np.atleast_2d(np.asarray(Pdat, dtype=float))
        Idat = np.asarray(Idat)
        Wdat = np.asarray(Wdat, dtype=float)
    if not anisotropicIntensities:
        F.logmsg(1, '  neglecting amplitude anisotropy, taking amplitude average')
        if not useEigenFields:
            Idat = np.mean(Idat) * np.ones(Idat.shape)
    nTransitions = 1 if useEigenFields else Pdat.shape[0]
    if crystalSample and not isempty(Exp.CrystalSymmetry):
        nSites = (len(Pdat) if useEigenFields else Pdat.size) / nTransitions / nOrientations
    else:
        nSites = 1
    if not FieldSweep and SweepAutoRange:
        minFreq, maxFreq = np.nanmin(Pdat), np.nanmax(Pdat)
        padding = (maxFreq - minFreq) / 5
        if padding == 0:
            padding = 0.1
        if anisotropicWidths:
            padding = max(padding, 5 * np.nanmax(Wdat))
        padding = max(padding, 5 * np.sum(lw))
        Exp.mwRange = np.array([[max(0.0, minFreq - padding), maxFreq + padding]])
        F.logmsg(1, '  automatic frequency range [%g %g] GHz', vec(Exp.mwRange)[0], vec(Exp.mwRange)[1])
    NaN_in_Pdat = np.any(np.isnan(Pdat), axis=1) if not useEigenFields else np.zeros(1, dtype=bool)
    if FieldSweep and Method != 6 and not useEigenFields:
        T_ = np.atleast_2d(np.asarray(Transitions))
        loopingTransitionsPresent = np.unique(T_, axis=0).shape[0] < T_.shape[0]
        if loopingTransitionsPresent and not crystalSample:
            F.logmsg(0, '** Looping transitions found. Artifacts at coalescence points possible.')
    else:
        loopingTransitionsPresent = False
    SweepRange = vec(np.asarray(Exp.Range if FieldSweep else Exp.mwRange, dtype=float))
    xAxis = linspace(SweepRange[0], SweepRange[1], nPoints)
    deltaX = xAxis[1] - xAxis[0]
    Exp.deltaX = deltaX

    # ---------------------------------------------------------------- spectrum construction
    F.logmsg(1, '-absorption spectrum construction----------------------')
    BruteForceSum = useEigenFields or _t(Opt.BruteForce)
    nOctants = int(_num(Opt.nOctants))
    axialGrid = nOctants == 0
    anisotropicSpectrum = nOctants >= 0
    if ImmediateBinning:
        pass
    elif not BruteForceSum:
        doInterpolation = anisotropicSpectrum and GridSize[1] > 1
        if doInterpolation:
            if axialGrid:
                InterpMode = ['L3', 'L3', 'L3'] if np.any(NaN_in_Pdat) else ['G3', 'L3', 'L3']
            else:
                InterpMode = ['L1', 'L1', 'L1'] if np.any(NaN_in_Pdat) else ['G3', 'L1', 'L1']
            msg = '  interpolation (factor %d, method %s/%s/%s)' % (GridSize[1], *InterpMode)
        else:
            GridSize[1] = 1
            Opt.GridSize = GridSize.reshape(1, -1)
            msg = '  interpolation off'
        nfKnots = (GridSize[0] - 1) * GridSize[1] + 1
        F.logmsg(1, msg)
        doProjection = anisotropicSpectrum and not anisotropicWidths
        doSummation = not doProjection
        if doSummation:
            useGaussianTemplate = True
            if not anisotropicWidths:
                if lw[0] > 0:
                    thisWid = lw[0]
                    lw[0] = 0
                    ConvolutionBroadening = lw[1] > 0
                else:
                    useGaussianTemplate = False
                    thisWid = lw[1]
                    lw[1] = 0
                    ConvolutionBroadening = False
                if not ConvolutionBroadening:
                    DerivHarmonic = DerivHarmonic + ConvHarmonic
                    ConvHarmonic = 0.0
                if crystalSample:
                    thisWid = np.full((nTransitions, 1), thisWid)
            if useGaussianTemplate:
                x0T = 5e4
                wT = x0T / 2.5
                xT = np.arange(0, 2 * x0T)
                if mwPhase != 0 and lw[1] == 0:
                    print('Exp.mwPhase is disregarded for Gaussian-only spectrum. Add a Lorentzian component.')
                Template = np.asarray(F.gaussian(xT, x0T, wT, -1.0))
            else:
                x0T = 1e5
                wT = x0T / 20
                xT = np.arange(0, 2 * x0T)
                Template = np.asarray(F.lorentzian(xT, x0T, wT, -1.0, mwPhase))
            Template = vec(Template)
        msg = ('triangle/segment projection' if doProjection else
               ('summation using Gaussian template' if useGaussianTemplate else 'summation using Lorentzian template'))
        F.logmsg(1, '  %s', msg)
        if separateOrientationSpectra:
            nSpectra, msg = nOrientations, '(separate orientations)'
        elif separateSiteSpectra:
            nSpectra, msg = int(nSites), '(separate sites)'
        elif separateTransitionSpectra:
            nSpectra, msg = nTransitions, '(separate transitions)'
        else:
            nSpectra, msg = 1, ''
        spec = np.zeros((nSpectra, nPoints))
        F.logmsg(1, '  spectrum array size: %dx%d %s', spec.shape[0], spec.shape[1], msg)
        OriWeights = vec(np.asarray(Exp.OriWeights, dtype=float))
        if crystalSample:
            if nTransitions > 0:
                if not anisotropicIntensities:
                    thisInt = np.ones((nTransitions, 1))
                iOriSite = 0
                spcidx = 0
                for iOri in range(nOrientations):
                    if separateSiteSpectra:
                        spcidx = 0
                    elif separateOrientationSpectra:
                        spcidx += 1
                    else:
                        spcidx = 1
                    for iSite in range(int(nSites)):
                        if separateSiteSpectra:
                            spcidx += 1
                        thisPos = Pdat[:, iOriSite]
                        if anisotropicIntensities:
                            thisInt = Idat[:, iOriSite]
                        if anisotropicWidths:
                            thisWid = Wdat[:, iOriSite]
                        thisspec = np.asarray(F.lisum1i(Template, x0T, wT, thisPos, thisInt, thisWid, xAxis))
                        thisspec = vec(thisspec) / nSites / nOrientations * (2 * np.pi) * OriWeights[iOri]
                        if not separateSiteSpectra and not separateOrientationSpectra:
                            spec = spec + thisspec
                        else:
                            spec[spcidx - 1, :] = spec[spcidx - 1, :] + thisspec
                        iOriSite += 1
        elif not anisotropicSpectrum:
            if not anisotropicIntensities:
                thisInt = 1.0
            spcidx = 0
            for iTrans in range(nTransitions):
                thisPos = Pdat[iTrans, :]
                if anisotropicIntensities:
                    thisInt = Idat[iTrans, :]
                if anisotropicWidths:
                    thisWid = Wdat[iTrans, :]
                thisspec = vec(np.asarray(F.lisum1i(Template, x0T, wT, thisPos, thisInt, thisWid, xAxis)))
                thisspec = (2 * np.pi) * thisspec * OriWeights[0]
                if not separateTransitionSpectra:
                    spec = spec + thisspec
                else:
                    spec[spcidx, :] = thisspec
                    spcidx += 1
        else:
            if axialGrid:
                if doInterpolation:
                    fgrid = F.sphgrid(0, nfKnots)
                    fgrid = fgrid[0] if isinstance(fgrid, tuple) else fgrid
                    fphi = np.asarray(fgrid.phi, dtype=float)
                    fthe = np.asarray(fgrid.theta, dtype=float)
                else:
                    fthe = np.asarray(Exp.theta, dtype=float)
                    fphi = np.asarray(Exp.phi, dtype=float)
                fWeights = -np.diff(vec(np.cos(fthe))) * 4 * np.pi
                if _t(Opt.partiallyOrderedSample):
                    raise ValueError('Cannot use axial grid for partially ordered samples.')
                F.logmsg(1, '  total %d segments, %d transitions', fthe.size - 1, nTransitions)
            else:
                if doInterpolation:
                    fgrid, tri = F.sphgrid(Opt.GridSymmetry, nfKnots, nargout=2)[:2]
                    fphi = np.asarray(fgrid.phi, dtype=float)
                    fthe = np.asarray(fgrid.theta, dtype=float)
                else:
                    tri = Exp.tri
                    fthe = np.asarray(Exp.theta, dtype=float)
                    fphi = np.asarray(Exp.phi, dtype=float)
                idxTri = np.asarray(tri.idx).T                      # 3 x nTri, 1-based
                fWeights = vec(np.asarray(tri.areas, dtype=float))
                if _t(Opt.partiallyOrderedSample):
                    fp, ft = vec(fphi), vec(fthe)
                    centerPhi = np.mean(fp[idxTri.astype(int) - 1], axis=0)
                    centerTheta = np.mean(ft[idxTri.astype(int) - 1], axis=0)
                    R0 = Opt.R_L2S[0] if isinstance(Opt.R_L2S, list) else Opt.R_L2S
                    orderingWeights = vec(np.asarray(F.orifun_M2L(Exp.Ordering, R0, centerPhi, centerTheta)))
                    if np.any(orderingWeights < 0):
                        raise ValueError('User-supplied orientation distribution gives negative values.')
                    if np.all(orderingWeights == 0):
                        raise ValueError('User-supplied orientation distribution is all-zero.')
                    fWeights = fWeights * orderingWeights / np.sum(orderingWeights)
                fWeights = 4 * np.pi * fWeights / np.sum(fWeights)
                F.logmsg(1, '  total %d triangles (%d orientations), %d transitions', idxTri.shape[1], fthe.size, nTransitions)
            if not anisotropicIntensities:
                fInt = np.ones(np.shape(fthe))
            if not anisotropicWidths:
                fWid = np.zeros(np.shape(fthe))
            minBroadening = np.inf
            nBroadenings = 0
            sumBroadenings = 0.0
            spcidx = 0
            for iTrans in range(nTransitions):
                if doInterpolation:
                    fPos = F.gridinterp(Pdat[iTrans, :].reshape(1, -1), Opt.GridParams, fphi, fthe, InterpMode[0])
                    if anisotropicIntensities:
                        fInt = F.gridinterp(Idat[iTrans, :].reshape(1, -1), Opt.GridParams, fphi, fthe, InterpMode[1])
                    if anisotropicWidths:
                        fWid = F.gridinterp(Wdat[iTrans, :].reshape(1, -1), Opt.GridParams, fphi, fthe, InterpMode[2])
                else:
                    fPos = Pdat[iTrans, :]
                    if anisotropicIntensities:
                        fInt = Idat[iTrans, :]
                    if anisotropicWidths:
                        fWid = Wdat[iTrans, :]
                fPos, fInt, fWid = (vec(np.asarray(a)) for a in (fPos, fInt, fWid))
                msg1 = ''
                if not nonEquiPops and np.any(np.real(fInt) < 0):
                    msg1 = 'intensities'
                if np.any(fWid < 0):
                    msg1 = 'widths'
                if msg1:
                    raise ValueError(f'Negative {msg1} encountered! Please report!')
                if doProjection:
                    if axialGrid:
                        thisspec = F.projectzones(fPos, fInt, fWeights, xAxis)
                    else:
                        thisspec = F.projecttriangles(idxTri, fWeights, fPos, fInt, xAxis)
                else:
                    if axialGrid:
                        fPosC = (fPos[:-1] + fPos[1:]) / 2
                        fIntC = fWeights * (fInt[:-1] + fInt[1:]) / 2
                        fSpread = np.abs(fPos[:-1] - fPos[1:])
                        fWidM = (fWid[:-1] + fWid[1:]) / 2
                        c1, c2 = 1.57246, 18.6348
                    else:
                        it = idxTri.astype(int) - 1
                        fPosSorted = np.sort(fPos[it], axis=0)
                        fPosC = np.mean(fPosSorted, axis=0)
                        fIntC = fWeights * np.mean(fInt[it], axis=0)
                        fSpread = fPosSorted[2, :] - fPosSorted[0, :]
                        fWidM = np.mean(fWid[it], axis=0)
                        c1, c2 = 2.8269, 42.6843
                    with np.errstate(divide='ignore', invalid='ignore'):
                        Lambda = fWidM / fSpread
                        gam = 1 / np.sqrt(c1 * Lambda ** 2 + c2 * Lambda ** 4)
                    gam[np.isinf(gam)] = 0
                    fWidC = fWidM * (1 + _num(Opt.Smoothing) * gam)
                    thisspec = F.lisum1i(Template, x0T, wT, fPosC, fIntC, fWidC, xAxis)
                    minBroadening = min(minBroadening, np.min(Lambda))
                    sumBroadenings += np.sum(Lambda)
                    nBroadenings += Lambda.size
                thisspec = vec(np.asarray(thisspec)) * (2 * np.pi)
                if not separateTransitionSpectra:
                    spec = spec + thisspec
                else:
                    spec[spcidx, :] = thisspec
                    spcidx += 1
            if doSummation and nBroadenings:
                F.logmsg(1, '  Smoothness: overall %0.4g, worst %0.4g\n   (<0.5: probably bad, 0.5-3: ok, >3: overdone)',
                         sumBroadenings / nBroadenings, minBroadening)
    else:
        F.logmsg(1, '  constructing stick spectrum (no interpolation)')
        F.logmsg(1, '  summation over %d orientations', nOrientations)
        if not isempty(Opt.separate):
            raise ValueError('Cannot return separate subspectra when constructing stick spectrum.')
        spec = np.zeros((1, nPoints))
        prefactor = (nPoints - 1) / (SweepRange[1] - SweepRange[0])
        OriWeights = vec(np.asarray(Exp.OriWeights, dtype=float))
        for iOri in range(nOrientations):
            if isinstance(Pdat, list):
                thisP, Amplitudes = vec(np.asarray(Pdat[iOri], dtype=float)), vec(np.asarray(Idat[iOri]))
            else:
                thisP, Amplitudes = Pdat[:, iOri], vec(Idat[:, iOri])
            idxPos = np.floor(1 + prefactor * (thisP - SweepRange[0]) + 0.5)   # MATLAB round (positive)
            keep = ~((idxPos < 1) | (idxPos > nPoints))
            idxPos, Amplitudes = idxPos[keep].astype(int), Amplitudes[keep]
            if anisotropicIntensities:
                np.add.at(spec[0], idxPos - 1, OriWeights[iOri] * Amplitudes)
            else:
                np.add.at(spec[0], idxPos - 1, OriWeights[iOri])
        spec = spec * (2 * np.pi) / deltaX

    # ---------------------------------------------------------------- final
    F.logmsg(1, '-final-------------------------------------------------')
    spec = np.atleast_2d(np.asarray(spec))
    if FieldSweep and not crystalSample and not summedOutput and loopingTransitionsPresent:
        T_ = np.atleast_2d(np.asarray(Transitions))
        Transitions, idx = np.unique(T_, axis=0, return_inverse=True)
        newspec = np.zeros((Transitions.shape[0], nPoints), dtype=spec.dtype)
        for k, i in enumerate(vec(idx)):
            newspec[i, :] += spec[k, :]
        spec = newspec
    if ConvolutionBroadening:
        F.logmsg(1, '  harmonic %d: using convolution', ConvHarmonic)
        fwhmG, fwhmL = lw[0], lw[1]
        if fwhmL > 0:
            HarmonicL, mwPhaseL, HarmonicG, mwPhaseG = ConvHarmonic, mwPhase, 0.0, 0.0
        else:
            HarmonicL, mwPhaseL, HarmonicG, mwPhaseG = 0.0, 0.0, ConvHarmonic, mwPhase
        unitstr = 'mT' if FieldSweep else 'GHz'
        exceedsLowerLimit = bool(np.any(spec[:, 0] != 0))
        exceedsHigherLimit = bool(np.any(spec[:, -1] != 0))
        if exceedsLowerLimit:
            F.logmsg(0, '** Spectrum exceeds sweep range. Artifacts at lower and upper limits possible.' if exceedsHigherLimit
                     else '** Spectrum exceeds sweep range. Artifacts at lower limit possible.')
        elif exceedsHigherLimit:
            F.logmsg(0, '** Spectrum exceeds sweep range. Artifacts at upper limit possible.')
        if exceedsLowerLimit or exceedsHigherLimit:
            nPad = int(np.floor(max(fwhmG, fwhmL) / deltaX * _num(Opt.PaddingMultiplier) + 0.5))
            spec = np.hstack([np.repeat(spec[:, :1], nPad, axis=1), spec, np.repeat(spec[:, -1:], nPad, axis=1)])
        else:
            nPad = 0
        if fwhmL != 0:
            if fwhmL > 2 * deltaX:
                F.logmsg(1, '  convoluting with Lorentzian, FWHM %g %s, derivative %d', fwhmL, unitstr, HarmonicL)
                fwhm = np.array([[0.0, fwhmL]]) if spec.shape[0] > 1 else fwhmL
                spec = np.atleast_2d(np.asarray(F.convspec(spec, deltaX, fwhm, HarmonicL, 0.0, mwPhaseL)))
            elif HarmonicL != 0:
                raise ValueError(f'Lorentzian linewidth (FWHM {fwhmL:g} {unitstr}) is smaller than 2 increments '
                                 f'(2x{deltaX:g} = {2 * deltaX:g} {unitstr}) - cannot do convolution.\nIncrease '
                                 'linewidth, or increment number of points in Exp.nPoints.')
        if fwhmG != 0:
            if fwhmG > 2 * deltaX:
                F.logmsg(1, '  convoluting with Gaussian, FWHM %g %s, derivative %d', fwhmG, unitstr, HarmonicG)
                fwhm = np.array([[0.0, fwhmG]]) if spec.shape[0] > 1 else fwhmG
                spec = np.atleast_2d(np.asarray(F.convspec(spec, deltaX, fwhm, HarmonicG, 1.0, mwPhaseG)))
            elif HarmonicG != 0:
                raise ValueError(f'Gaussian linewidth (FWHM {fwhmG:g} {unitstr}) is smaller than 2 increments '
                                 f'(2x{deltaX:g} = {2 * deltaX:g} {unitstr}) - cannot do convolution.\nIncrease '
                                 'linewidth, increment number of points in Exp.nPoints.')
        if nPad > 0:
            spec = spec[:, nPad:nPad + nPoints]
    elif DerivHarmonic > 0:
        F.logmsg(1, '  harmonic %d: using differentiation', DerivHarmonic)
        for _ in range(int(DerivHarmonic)):
            dspec = np.diff(spec, axis=1) / deltaX
            spec = (dspec[:, np.r_[0, 0:dspec.shape[1]]] + dspec[:, np.r_[0:dspec.shape[1], dspec.shape[1] - 1]]) / 2
    else:
        F.logmsg(1, '  harmonic 0: no differentiation')
    if FieldSweep and ModAmp > 0:
        F.logmsg(1, '  applying field modulation')
        if summedOutput:
            spec = np.atleast_2d(np.asarray(F.fieldmod(xAxis, spec, ModAmp, ModHarmonic)))
        else:
            for iSpec in range(spec.shape[0]):
                spec[iSpec, :] = vec(np.asarray(F.fieldmod(xAxis, spec[iSpec, :], ModAmp, ModHarmonic)))

    info = Struct(Transitions=Transitions, nSites=float(nSites), nOrientations=float(nOrientations))
    F.logmsg(1, '=end=pepper======================================\n')
    config.LogLevel = None   # MATLAB: clear global EasySpinLogLevel
    xAxis = xAxis.reshape(1, -1)
    if plot or nargout == 0:
        F.cwepr_plot(xAxis, spec, Exp)
        if nargout == 0:
            return None
    if nargout == 1:
        return spec
    if nargout == 2:
        return xAxis, spec
    return xAxis, spec, info
