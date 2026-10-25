"""ENDOR spectra simulation"""
import datetime
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


def _isnan(x):
    return np.isnan(np.asarray(x, dtype=float))


def _mround(x):
    """MATLAB round (half away from zero)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _mand(*xs):
    """MATLAB elementwise & with scalar expansion; any empty operand gives a 0x0 result."""
    arrs = [np.asarray(x) for x in xs]
    if any(a.size == 0 for a in arrs):
        return np.zeros((0, 0), dtype=bool)
    out = arrs[0] != 0
    for a in arrs[1:]:
        out = out & (a != 0)
    return bool(out) if out.size == 1 else out


def _firstdim(a):
    a = np.asarray(a)
    for k, s in enumerate(a.shape):
        if s != 1:
            return k
    return 0


def _mtimes(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size == 1 or b.size == 1:
        return a * b
    return a2(a) @ a2(b)


def _salt_plot(xAxis, spec, Exp):
    import matplotlib.pyplot as plt
    plt.cla()
    plt.plot(np.ravel(xAxis), np.atleast_2d(spec).T)
    plt.axis('tight')
    plt.xlabel('frequency (MHz)')
    plt.ylabel('intensity (arb.u.)')
    if isfield(Exp, 'mwFreq') and not _if(_isnan(Exp.mwFreq)):
        plt.title('ENDOR at %0.8g GHz and %0.8g mT' % (np.ravel(Exp.mwFreq)[0], np.ravel(Exp.Field)[0]))
    else:
        plt.title('ENDOR at %0.8g mT' % np.ravel(Exp.Field)[0])


@nout(-1)
def salt(Sys=None, Exp=None, Opt=None, nargout=None, plot=False):
    """salt(Sys, Exp, Opt) -> (x, spec); nargout=1 -> spec, nargout=3 -> (x, spec, info).
    ENDOR powder/crystal spectra (endorfrq or endorfrq_perturb, projection or summation).
    plot=True plots the result (MATLAB's nargout=0 behaviour)."""
    if nargout is None:
        nargout = 2
    StartTime = datetime.datetime.now()
    if Sys is None:
        raise ValueError('Please supply a spin system as first input argument.')
    if Exp is None:
        raise ValueError('Please supply experimental parameters as second input argument.')
    if nargout > 3:
        raise ValueError('Too many output arguments.')
    if Opt is None or isempty(Opt):
        Opt = Struct()
    if not isinstance(Sys, (dict, list)):
        raise ValueError('The first input (Sys) must be a structure or a cell array of structures.')
    if not isinstance(Exp, dict):
        raise ValueError('The second input (Exp) must be a structure.')
    if not isinstance(Opt, dict):
        raise ValueError('The third input (Opt) must be a structure.')
    if isinstance(Sys, dict):
        Sys = Struct(Sys)
    Exp = Struct(Exp)
    Opt = Struct(Opt)

    if not isfield(Opt, 'Verbosity'):
        Opt.Verbosity = 0
    config.LogLevel = Opt.Verbosity

    SweepAutoRange = (not isfield(Exp, 'Range') or isempty(Exp.Range)) and \
        (not isfield(Exp, 'CenterSweep') or isempty(Exp.CenterSweep))

    if not isfield(Opt, 'IsoCutoff'):
        Opt.IsoCutoff = 1e-4
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

    singleIsotopologue = isfield(Sys, 'singleiso') and _if(Sys.singleiso)
    if not singleIsotopologue:
        thirdOutput = nargout >= 3
        xAxis, spec, info = F.compisoloop(F.salt, Sys, Exp, Opt, SweepAutoRange, thirdOutput,
                                          separateComponentSpectra)
        if plot:
            _salt_plot(xAxis, spec, Exp)
        if nargout == 0:
            return None
        if nargout == 1:
            return spec
        if nargout == 2:
            return xAxis, spec
        return xAxis, spec, info

    EasySpinLogLevel = config.LogLevel
    F.logmsg(1, '=begin=salt=======%s=================', StartTime.strftime('%d-%b-%Y %H:%M:%S'))
    F.logmsg(2, '  log level %d', EasySpinLogLevel)
    F.logmsg(1, '-general-----------------------------------------------')

    Sys, err = F.validatespinsys(Sys)
    if not isempty(err):
        raise ValueError(err)
    Sys = Struct(Sys)
    if _if(Sys.MO_present):
        raise ValueError('salt does not support Sys.Ham* parameters.')
    if np.any(a2(Sys.get('L', 0)) != 0):
        raise ValueError('salt does not support Sys.L.')
    if np.any(a2(Sys.n) > 1):
        raise ValueError('salt does not support sets of equivalent nuclei (Sys.n>1).')

    lwEndor = vec(a2(Sys.lwEndor).astype(float))
    ConvolutionBroadening = bool(np.any(lwEndor > 0))
    F.logmsg(1, '  system with %d electron spin(s), %d nuclear spin(s), total %d states',
             int(Sys.nElectrons), int(Sys.nNuclei), int(Sys.nStates))

    if isfield(Sys, 'ExciteWidth'):
        raise ValueError('You gave Sys.ExciteWidth, but it should be Exp.ExciteWidth. Please correct.')

    if not isfield(Opt, 'Method'):
        Opt.Method = 'matrix'
    Method = F.parseoption(Opt, 'Method', ['matrix', 'perturb', 'perturb1', 'perturb2'], nargout=1)
    if isinstance(Method, tuple):
        Method = Method[0]
    if Method == 1:
        Opt.PerturbOrder = 0
        Method = 1
    elif Method == 2:
        Opt.PerturbOrder = 1
        Method = 2
    elif Method == 3:
        Opt.PerturbOrder = 1
        Method = 2
    elif Method == 4:
        Opt.PerturbOrder = 2
        Method = 2

    DefaultExp = Struct()
    DefaultExp.Field = np.nan
    DefaultExp.mwFreq = np.nan
    DefaultExp.Range = np.nan
    DefaultExp.CenterSweep = np.nan
    DefaultExp.nPoints = 1024
    DefaultExp.Temperature = np.nan
    DefaultExp.ExciteWidth = np.inf
    DefaultExp.Ordering = np.zeros((0, 0))
    DefaultExp.SampleRotation = np.zeros((0, 0))
    DefaultExp.SampleFrame = np.zeros((0, 0))
    DefaultExp.CrystalSymmetry = ''
    DefaultExp.MolFrame = np.zeros((0, 0))
    DefaultExp.Harmonic = 0
    Exp = F.adddefaults(Exp, DefaultExp)

    if isfield(Exp, 'Orientations'):
        raise ValueError('Exp.Orientations is no longer supported, use Exp.SampleFrame instead.')
    if isfield(Exp, 'CrystalOrientation'):
        raise ValueError('Exp.CrystalOrientation is no longer supported, use Exp.SampleFrame instead.')
    if isfield(Exp, 'lightBeam') and not isempty(Exp.lightBeam):
        raise ValueError('Photoselection (via Exp.lightBeam) is not supported.')

    if _if(_isnan(Exp.Field)):
        raise ValueError('Experiment.Field is missing!')
    if _if(_isnan(Exp.mwFreq)):
        if not _if(np.isinf(np.asarray(Exp.ExciteWidth, dtype=float))):
            raise ValueError('Experiment.mwFreq is missing! It is needed, since Exp.ExciteWidth is given.')

    AutoRange = _if(_mand(_isnan(Exp.Range), _isnan(Exp.CenterSweep)))

    if _if(~_isnan(Exp.CenterSweep)):
        if _if(~_isnan(Exp.Range)):
            F.logmsg(0, 'Using Experiment.CenterSweep and ignoring Experiment.Range.')
        CS = vec(np.asarray(Exp.CenterSweep, dtype=float))
        if CS[1] <= 0:
            raise ValueError('Sweep range in Exp.CenterSweep must be positive.')
        Exp.Range = CS[0] + np.array([[-1.0, 1.0]]) * CS[1] / 2
        if Exp.Range[0, 0] < 0:
            raise ValueError('Start value resulting from Exp.CenterSweep is negative (%g).' % Exp.Range[0, 0])

    if _if(~_isnan(Exp.Range)):
        R = np.asarray(Exp.Range)
        Rv = vec(R)
        if np.any(np.diff(Rv) <= 0) or np.any(~np.isfinite(Rv)) or np.iscomplexobj(R) or np.any(Rv < 0) \
                or Rv.size != 2:
            raise ValueError('Experiment.Range is not valid!')

    g = a2(Sys.g).astype(float)
    if np.size(Exp.mwFreq) == 1 and _if(np.asarray(Exp.mwFreq) == 0) and np.all(np.diff(g, axis=_firstdim(g)) == 0):
        Exp.mwFreq = np.mean(g, axis=_firstdim(g)) * _c('bmagn') * Exp.Field * 1e-3 / _c('planck') / 1e9
        if np.size(Exp.mwFreq) == 1:
            Exp.mwFreq = float(np.ravel(Exp.mwFreq)[0])

    Exp, Opt = F.p_sampletype(Exp, Opt)
    Exp, Opt = Struct(Exp), Struct(Opt)
    crystalSample = _if(Opt.crystalSample)
    if _if(Opt.partiallyOrderedSample):
        raise ValueError('Partially ordered samples are not implemented in salt.')

    if EasySpinLogLevel is not None and EasySpinLogLevel >= 1:
        Rv = vec(np.asarray(Exp.Range, dtype=float))
        if not AutoRange:
            msg = 'field %g mT, rf range [%g %g] MHz, %d points' % (np.ravel(Exp.Field)[0], Rv[0], Rv[-1],
                                                                    np.ravel(Exp.nPoints)[0])
        else:
            msg = 'field %g mT, automatic rf range, %d points' % (np.ravel(Exp.Field)[0], np.ravel(Exp.nPoints)[0])
        if not _if(_isnan(Exp.mwFreq)):
            msg = 'mw %g GHz, %s' % (np.ravel(Exp.mwFreq)[0], msg)
        else:
            msg = 'mw not given, %s' % msg
        F.logmsg(1, '  ' + msg)

    nonEquiPops = False
    if np.all(np.isfinite(np.asarray(Exp.Temperature, dtype=float))):
        msg = '  temperature %g K' % np.ravel(Exp.Temperature)[0]
    else:
        msg = '  no temperature'
    F.logmsg(1, msg)

    if _if(np.asarray(Exp.Harmonic) > 0) and np.all(lwEndor == 0):
        # EasySpin's logmsg rejects level -Inf; the port's logmsg logs it as a warning (docs/DEVIATIONS.md).
        F.logmsg(-np.inf, 'WARNING: Cannot compute derivative of a stick spectrum! Returning absorption spectrum.')
        F.logmsg(-np.inf, 'WARNING:   Add a line width (strain or convolution) or set Harmonic=0.')
        Exp.Harmonic = 0

    if isfield(Exp, 'HStrain'):
        raise ValueError('You gave Exp.HStrain, but it should be Sys.HStrain. Please correct.')

    DefaultOpt = Struct()
    if Method == 2:
        DefaultOpt.GridSize = np.array([[61.0, 0.0]])
    else:
        DefaultOpt.GridSize = np.array([[31.0, 3.0]])
    DefaultOpt.GridSymmetry = ''
    DefaultOpt.GridFrame = np.zeros((0, 0))
    DefaultOpt.separate = ''
    DefaultOpt.Smoothing = 2
    DefaultOpt.minEffKnots = 5
    DefaultOpt.OriWeights = np.zeros((0, 0))
    DefaultOpt.OriThreshold = 1e-4
    DefaultOpt.IncludeOriWeights = 1
    DefaultOpt.OriPreSelect = np.zeros((0, 0))
    DefaultOpt.PaddingMultiplier = 3
    Opt = F.adddefaults(Opt, DefaultOpt)

    GridSize = vec(np.asarray(Opt.GridSize, dtype=float))
    if GridSize[0] < 2:
        raise ValueError('Opt.GridSize(1) must be 2 or larger.')
    if GridSize.size < 2:
        GridSize = np.array([GridSize[0], DefaultOpt.GridSize[0, 1]])
        Opt.GridSize = GridSize.reshape(1, -1)

    if not crystalSample:
        if separateOrientationSpectra:
            raise ValueError("\nCannot return separate orientations for non-crystal spectra "
                             "(Opt.separate='orientations').\nUse other setting for Opt.separate.\n")
        if separateSiteSpectra:
            raise ValueError("\nCannot return separate sites for non-crystal spectra (Opt.separate='sites').\n"
                             "Use other setting for Opt.separate.\n")
    else:
        if separateTransitionSpectra:
            raise ValueError("\n  Cannot return separate transitions for crystal spectra "
                             "(Opt.separate='transitions').\n  Use other setting for Opt.separate.\n")

    suppliedOriWeights = not isempty(Opt.OriWeights)
    if suppliedOriWeights:
        Opt.OriWeights = vec(np.asarray(Opt.OriWeights, dtype=float)).reshape(1, -1)

    if np.any(np.asarray(Opt.OriPreSelect) != 0) and suppliedOriWeights:
        raise ValueError('You cannot use OriPreSelect and supply OriWeights simultaneously!')

    if isfield(Opt, 'nKnots'):
        raise ValueError('Options.nKnots is obsolete. Use Options.GridSize instead, e.g. Options.GridSize = 91.')
    if isfield(Opt, 'Symmetry'):
        raise ValueError("Options.Symmetry is obsolete. Use Options.GridSymmetry instead, e.g. "
                         "Options.GridSymmetry = 'D2h'.")
    if isfield(Opt, 'Convolution'):
        raise ValueError('Options.Convolution is obsolete! Please remove from code!')
    if isfield(Opt, 'Width'):
        raise ValueError('Options.Width is obsolete! Please remove from code.')
    if isfield(Opt, 'nSpline'):
        raise ValueError('Options.nSpline is obsolete. Use a second number in Options.GridSize instead, e.g. '
                         'Options.GridSize = [19 5] for Options.nSpline = 5.')
    if isfield(Opt, 'LineShape'):
        raise ValueError('Options.LineShape is obsolete. Use System.lwEndor instead.')
    if isfield(Opt, 'Output'):
        raise ValueError('Options.Output is obsolete. Use Options.separate instead.')

    nPoints = int(np.ravel(Exp.nPoints)[0])

    def _finish(xAxis, spec, Transitions):
        if plot:
            _salt_plot(xAxis, spec, Exp)
        if nargout == 1:
            out = spec
        elif nargout == 2:
            out = (xAxis, spec)
        elif nargout == 3:
            info = Struct()
            info.Transitions = Transitions
            out = (xAxis, spec, info)
        else:
            out = None
        return out

    if int(Sys.nNuclei) == 0:
        Rv = vec(np.asarray(Exp.Range, dtype=float))
        Exp.deltaX = np.diff(Rv) / (nPoints - 1)
        xAxis = (Rv[0] + np.arange(nPoints) * Exp.deltaX).reshape(1, -1)
        spec = np.zeros((1, nPoints))
        Transitions = np.zeros((0, 0))
        out = _finish(xAxis, spec, Transitions)
        return out

    Exp, Opt, nOrientations = F.p_gridsetup(Sys, Exp, Opt)
    Exp, Opt = Struct(Exp), Struct(Opt)
    nOrientations = int(nOrientations)
    GridSize = vec(np.asarray(Opt.GridSize, dtype=float)).copy()

    Opt.DoPreSelection = _mand(not crystalSample, Opt.OriPreSelect, np.asarray(Opt.OriThreshold) > 0,
                               isempty(Opt.OriWeights))
    if _if(Opt.DoPreSelection):
        F.logmsg(1, '  computing orientations which fall into excitation window')
        OriSys = F.anisosubsys(Sys)
        OriExp = Struct(Exp)
        OriExp.Orientations = np.vstack([a2(Exp.phi), a2(Exp.theta)])
        Opt.OriWeights = np.asarray(F.orisel(OriSys, OriExp), dtype=float)
        Opt.OriWeights = Opt.OriWeights / np.max(Opt.OriWeights)
        F.logmsg(1, '  %g percent of all orientations above relative threshold %g'
                 % (100 * np.sum(Opt.OriWeights > Opt.OriThreshold) / nOrientations, Opt.OriThreshold))

    F.logmsg(1, '-resonances--------------------------------------------')
    MethodName = ['  method: matrix diagonalization', '  method: perturbation theory']
    F.logmsg(1, MethodName[Method - 1])
    F.logmsg(2, '  -endorfrq start-----------------------------------')
    if Method == 1:
        Pdat, Idat, Transitions, Info = F.endorfrq(Sys, Exp, Opt)
    else:
        Pdat, Idat, Transitions, Info = F.endorfrq_perturb(Sys, Exp, Opt)
    F.logmsg(2, '  -endorfrq end-------------------------------------')

    Pdat = a2(np.asarray(Pdat, dtype=float))
    Idat = np.asarray(Idat, dtype=float)
    if Idat.size > 1:
        Idat = a2(Idat)
    nTransitions = Pdat.shape[0]
    anisotropicWidths = False

    Rv = vec(np.asarray(Exp.Range, dtype=float))
    if nTransitions == 0:
        raise ValueError(('No ENDOR resonances between %g and %g MHz (%g mT).\n'
                          'Check frequency range, magnetic field and transition selection.')
                         % (Rv[0], Rv[-1], np.ravel(Exp.Field)[0]))
    F.logmsg(1, '  %d transitions with resonances in range', nTransitions)
    F.logmsg(2, '  positions min %g MHz, max %g MHz', np.nanmin(Pdat), np.nanmax(Pdat))
    anisotropicIntensities = Idat.size > 1
    if anisotropicIntensities:
        F.logmsg(2, '  amplitudes min %g, max %g', np.nanmin(Idat), np.nanmax(Idat))

    if suppliedOriWeights:
        F.logmsg(1, '  user supplied OriWeights: including them in intensity')
        if _if(Opt.IncludeOriWeights):
            Idat = Idat * np.asarray(Opt.OriWeights, dtype=float).reshape(1, -1)

    if crystalSample and not isempty(Exp.CrystalSymmetry):
        nSites = int(round(Pdat.size / nTransitions / nOrientations))
    else:
        nSites = 1

    if AutoRange:
        lo, hi = np.nanmin(Pdat), np.nanmax(Pdat)
        Range = np.array([lo, hi]) + np.array([-1.0, 1.0]) * np.max(lwEndor) * 2
        Range = Range + (Range[1] - Range[0]) * np.array([-1.0, 1.0]) * 0.2
        Range[0] = max(Range[0], 0)
        Exp.Range = Range.reshape(1, -1)

    Rv = vec(np.asarray(Exp.Range, dtype=float))
    deltaX = (Rv[1] - Rv[0]) / (nPoints - 1)
    Exp.deltaX = deltaX
    xAxis = (Rv[0] + np.arange(nPoints) * deltaX).reshape(1, -1)

    Selectivity = float(np.ravel(Info.Selectivity)[0]) if isfield(Info, 'Selectivity') else 0.0
    if Selectivity > 0:
        F.logmsg(1, '  orientation selection: %g (<1 very weak, 1 weak, 10 strong, >10 very strong)', Selectivity)
        GridTooCoarse = (GridSize[0] / float(Opt.minEffKnots) < Selectivity)
        if GridTooCoarse and not crystalSample:
            print('  ** Warning: Strong orientation selection ********************************')
            print('  Only %0.1f orientations within excitation window.' % (GridSize[0] / Selectivity))
            print('  Spectrum might be inaccurate!')
            print('  To remedy, do one (or both) of the following:')
            print('  - Increase Opt.GridSize (currently %d)' % GridSize[0])
            print('  - Increase Exp.ExciteWidth (currently %g MHz)' % np.ravel(Exp.ExciteWidth)[0])
            print('  *************************************************************************')
    else:
        F.logmsg(1, '  orientation selection: none')

    F.logmsg(1, '-absorption spectrum construction----------------------')
    NaN_in_Pdat = np.any(np.isnan(Pdat), axis=1)
    if np.any(NaN_in_Pdat):
        GridSize[1] = 1
        Opt.GridSize = GridSize.reshape(1, -1)

    nOctants = float(np.ravel(Opt.nOctants)[0])
    axialGrid = nOctants == 0
    usingGrid = nOctants >= 0

    doProjection = usingGrid and not anisotropicWidths
    Template = None
    if doProjection:
        msg = 'triangle/segment projection'
    else:
        msg = 'summation'
        Template = Struct()
        Template.x = 5e4
        Template.lw = Template.x / 2.5
        Template.y = F.gaussian(np.arange(0, 2 * Template.x).reshape(1, -1), Template.x, Template.lw, -1)
    F.logmsg(1, '  %s', msg)

    doInterpolation = usingGrid and GridSize[1] > 1
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

    msg = ''
    if separateOrientationSpectra:
        nSpectra = nOrientations
        msg = '(separate orientations)'
    elif separateSiteSpectra:
        nSpectra = nSites
        msg = '(separate sites)'
    elif separateTransitionSpectra:
        nSpectra = nTransitions
        msg = '(separate transitions)'
    else:
        nSpectra = 1
    spec = np.zeros((nSpectra, nPoints))
    F.logmsg(1, '  spectrum array size: %dx%d %s', spec.shape[0], spec.shape[1], msg)

    OriWeights = np.asarray(Exp.OriWeights, dtype=float) if isfield(Exp, 'OriWeights') else None
    if crystalSample:
        if Template is None:
            raise ValueError("Unrecognized function or variable 'Template'.")
        thisInt = np.ones(nTransitions)
        thisWid = np.zeros(nTransitions)
        OWv = vec(OriWeights)
        PdatF = Pdat.reshape(nTransitions, -1, order='F')
        IdatF = Idat.reshape(nTransitions, -1, order='F') if anisotropicIntensities else None
        iOriSite = 0
        spcidx = 0
        for iOri in range(nOrientations):
            if separateSiteSpectra:
                spcidx = 0
            elif separateOrientationSpectra:
                spcidx = spcidx + 1
            else:
                spcidx = 1
            for iSite in range(nSites):
                if separateSiteSpectra:
                    spcidx = spcidx + 1
                thisPos = PdatF[:, iOriSite]
                if anisotropicIntensities:
                    thisInt = IdatF[:, iOriSite]
                thisspec = F.lisum1i(Template.y, Template.x, Template.lw, thisPos, thisInt, thisWid, xAxis)
                thisspec = np.asarray(thisspec).reshape(1, -1)
                thisspec = thisspec / nSites / nOrientations
                thisspec = (2 * np.pi) * thisspec
                thisspec = OWv[iOri] * thisspec
                if not separateSiteSpectra and not separateOrientationSpectra:
                    spec = spec + thisspec
                else:
                    spec[spcidx - 1, :] = spec[spcidx - 1, :] + thisspec[0]
                iOriSite = iOriSite + 1
    elif not usingGrid:
        if Template is None:
            raise ValueError("Unrecognized function or variable 'Template'.")
        if not anisotropicIntensities:
            thisInt = Idat
        thisWid = 0
        spcidx = 0
        for iTrans in range(nTransitions):
            thisPos = Pdat[iTrans, :]
            if anisotropicIntensities:
                thisInt = Idat[iTrans, :]
            thisspec = F.lisum1i(Template.y, Template.x, Template.lw, thisPos, thisInt, thisWid, xAxis)
            thisspec = np.asarray(thisspec).reshape(1, -1) * (2 * np.pi)
            thisspec = _mtimes(OriWeights, thisspec).reshape(1, -1)
            if not separateTransitionSpectra:
                spec = spec + thisspec
            else:
                spcidx = spcidx + 1
                spec[spcidx - 1, :] = thisspec[0]
    else:
        if axialGrid:
            if doInterpolation:
                grid = F.sphgrid(Opt.GridSymmetry, nfKnots, nargout=1)
                if isinstance(grid, tuple):
                    grid = grid[0]
                fthe = vec(grid.theta)
                fphi = vec(grid.phi)
            else:
                fthe = vec(Exp.theta)
            fSegWeights = -np.diff(np.cos(fthe)) * 4 * np.pi
            F.logmsg(1, '  total %d segments, %d transitions', fthe.size - 1, nTransitions)
        else:
            if doInterpolation:
                grid, tri = F.sphgrid(Opt.GridSymmetry, nfKnots, nargout=2)
                fphi = vec(grid.phi)
                fthe = vec(grid.theta)
            else:
                tri = Exp.tri
                fthe = vec(Exp.theta)
                fphi = vec(Exp.phi)
            idxTri = np.asarray(tri.idx).T
            Areas = tri.areas
            F.logmsg(1, '  total %d triangles (%d orientations), %d transitions', idxTri.shape[1], fthe.size,
                     nTransitions)

        fInt = np.ones(fthe.shape)
        fWid = np.zeros(fthe.shape)
        minBroadening = np.inf
        nBroadenings = 0
        sumBroadenings = 0.0
        spcidx = 0
        for iTrans in range(nTransitions):
            if doInterpolation:
                fPos = F.gridinterp(Pdat[iTrans, :].reshape(1, -1), Opt.GridParams, fphi.reshape(1, -1),
                                    fthe.reshape(1, -1), InterpMode[0])
                if anisotropicIntensities:
                    fInt = F.gridinterp(Idat[iTrans, :].reshape(1, -1), Opt.GridParams, fphi.reshape(1, -1),
                                        fthe.reshape(1, -1), InterpMode[1])
            else:
                fPos = Pdat[iTrans, :]
                if anisotropicIntensities:
                    fInt = Idat[iTrans, :]
            fPos = vec(np.asarray(fPos, dtype=float))
            fInt = vec(np.asarray(fInt, dtype=float)).copy()

            msg1 = ''
            if not nonEquiPops and np.any(fInt < 0):
                maxInt = np.max(fInt)
                fInt[(fInt < 0) & (fInt / maxInt > -1e-6)] = 0
                if np.any(fInt < 0):
                    msg1 = 'intensities'
            if np.any(fWid < 0):
                msg1 = 'widths'
            if msg1:
                raise ValueError('Negative %s encountered! Please report!' % msg1)

            if doProjection:
                if axialGrid:
                    thisspec = F.projectzones(fPos.reshape(1, -1), fInt.reshape(1, -1), fSegWeights.reshape(1, -1),
                                              xAxis)
                else:
                    thisspec = F.projecttriangles(idxTri, Areas, fPos.reshape(1, -1), fInt.reshape(1, -1), xAxis)
            else:
                if axialGrid:
                    fPosC = (fPos[:-1] + fPos[1:]) / 2
                    fIntC = fSegWeights * (fInt[:-1] + fInt[1:]) / 2
                    fSpread = np.abs(fPos[:-1] - fPos[1:])
                    fWidM = (fWid[:-1] + fWid[1:]) / 2
                    c1, c2 = 1.57246, 18.6348
                else:
                    it = np.asarray(idxTri, dtype=int) - 1
                    fPosSorted = np.sort(fPos[it], axis=0)
                    fPosC = np.mean(fPosSorted, axis=0)
                    fIntC = vec(Areas) * np.mean(fInt[it], axis=0)
                    fSpread = fPosSorted[2, :] - fPosSorted[0, :]
                    fWidM = np.mean(fWid[it], axis=0)
                    c1, c2 = 2.8269, 42.6843
                Lambda = fWidM / fSpread
                gam = 1 / np.sqrt(c1 * Lambda ** 2 + c2 * Lambda ** 4)
                fWidC = fWidM * (1 + Opt.Smoothing * gam)
                thisspec = F.lisum1i(Template.y, Template.x, Template.lw, fPosC, fIntC, fWidC, xAxis)
                minBroadening = min(minBroadening, np.min(Lambda))
                sumBroadenings = sumBroadenings + np.sum(Lambda)
                nBroadenings = nBroadenings + Lambda.size
            thisspec = np.asarray(thisspec, dtype=float).reshape(1, -1) * (2 * np.pi)
            if not separateTransitionSpectra:
                spec = spec + thisspec
            else:
                spcidx = spcidx + 1
                spec[spcidx - 1, :] = thisspec[0]
        if not doProjection:
            F.logmsg(1, '  Smoothness: overall %0.4g, worst %0.4g\n   (<0.5: probably bad, 0.5-3: ok, >3: overdone)',
                     sumBroadenings / nBroadenings, minBroadening)

    F.logmsg(1, '-final-------------------------------------------------')
    Harmonic = int(np.ravel(Exp.Harmonic)[0])
    F.logmsg(1, '  harmonic %d', Harmonic)
    if ConvolutionBroadening:
        fwhmG = lwEndor[0]
        fwhmL = lwEndor[1]
        if fwhmL > 0:
            HarmonicL = Harmonic
            HarmonicG = 0
        else:
            HarmonicL = 0
            HarmonicG = Harmonic
        RangePadding = True
        if RangePadding:
            exceedsLowerLimit = bool(np.any(spec[:, 0] != 0))
            exceedsHigherLimit = bool(np.any(spec[:, -1] != 0))
            if exceedsLowerLimit:
                if exceedsHigherLimit:
                    F.logmsg(0, '** Spectrum exceeds frequency range. Artifacts at lower and upper frequency limits possible.')
                else:
                    F.logmsg(0, '** Spectrum exceeds frequency range. Artifacts at lower frequency limit possible.')
            else:
                if exceedsHigherLimit:
                    F.logmsg(0, '** Spectrum exceeds frequency range. Artifacts at upper frequency limit possible.')
            if exceedsLowerLimit or exceedsHigherLimit:
                nPad = int(_mround(max(fwhmG, fwhmL) / deltaX * float(Opt.PaddingMultiplier)))
                spec = np.hstack([np.tile(spec[:, [0]], (1, nPad)), spec])
                spec = np.hstack([spec, np.tile(spec[:, [-1]], (1, nPad))])
            else:
                nPad = 0
        if fwhmL != 0:
            if fwhmL > 2 * deltaX:
                F.logmsg(1, '  convoluting with Lorentzian, FWHM %g MHz, derivative %d', fwhmL, HarmonicL)
                fwhm = [fwhmL, 0] if min(spec.shape) == 1 else [0, fwhmL]
                spec = F.convspec(spec, deltaX, np.array([fwhm], dtype=float), HarmonicL, 0)
            else:
                if HarmonicL != 0:
                    raise ValueError('Lorentzian linewidth is smaller than increment - cannot perform convolution.')
        if fwhmG != 0:
            if fwhmG > 2 * deltaX:
                F.logmsg(1, '  convoluting with Gaussian, FWHM %g MHz, derivative %d', fwhmG, HarmonicG)
                fwhm = [fwhmG, 0] if min(spec.shape) == 1 else [0, fwhmG]
                spec = F.convspec(spec, deltaX, np.array([fwhm], dtype=float), HarmonicG, 1)
            else:
                if HarmonicG != 0:
                    raise ValueError('Gaussian linewidth is smaller than increment - cannot perform convolution.')
        spec = np.asarray(spec)
        if spec.ndim == 1:
            spec = spec.reshape(1, -1)
        if RangePadding:
            if nPad > 0:
                spec = spec[:, nPad:]
                spec = spec[:, :nPoints]
    else:
        F.logmsg(1, '  no linewidth given: returning stick spectrum')
        if Harmonic > 0:
            F.logmsg(1, '  harmonic %d: using differentiation', Harmonic)
            for h in range(Harmonic):
                dspec = np.diff(spec, axis=1) / deltaX
                spec = (np.hstack([dspec[:, [0]], dspec]) + np.hstack([dspec, dspec[:, [-1]]])) / 2
        else:
            F.logmsg(1, '  harmonic 0: returning absorption spectrum')

    out = _finish(xAxis, spec, Transitions)
    hmsString = F.elapsedtime(StartTime, datetime.datetime.now())
    if isinstance(hmsString, tuple):
        hmsString = hmsString[0]
    F.logmsg(1, 'salt took ' + hmsString)
    F.logmsg(1, '=end=salt=========%s=================\n', datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S'))
    config.LogLevel = None
    return out
