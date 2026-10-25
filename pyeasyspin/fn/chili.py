"""Simulation of cw EPR spectra in the slow motional regime"""
import inspect
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .._registry import F, nout
from .._struct import Struct, isfield
from .._mat import vec, isempty
from .._matlab import linspace as _linspace, bicgstab as _bicgstab   # noqa: F401
from ..config import config

# Opt.Diagnostics: MATLAB assigns the diagnostics struct to a variable of that name in the base
# workspace; here it is stored under that name in this dict.
DIAGNOSTICS = {}

# EasySpin divides the summed spectrum by the scale of the LAST Liouvillian when Opt.Rescale is on,
# so every other orientation and every other field point of an explicit field sweep keeps a factor
# scale_i/scale_last (DEVIATIONS.md). The port divides each solve by its own scale. Set this to
# True to reproduce EasySpin's numbers (docs/DEVIATIONS.md).
EASYSPIN_RESCALE = False

# EasySpin passes the director azimuth phi (rotation about the field axis) into the Liouvillian.
# The fast path's chili_lm mex reads only the real part of d2psi = wignerd(2,phi,theta,0), and the
# K-symmetrized general path gives a wrong Liouvillian for complex Wigner D as well. Both give
# phi-dependent spectra (MATLAB: 70 % and 10x of peak at phi = 0.9), although a rotation about the
# field axis cannot change the cw spectrum: the LMK-basis calculation (Opt.useLMKbasis, direct
# solver) is phi-invariant to 5e-16. The port uses phi = 0 in the Liouvillian (all paths;
# exact by that invariance; DEVIATIONS.md); True reproduces EasySpin.
EASYSPIN_DIRECTOR_PHI = False


def _c(name):
    v = getattr(F, name)()
    return v[0] if isinstance(v, tuple) else v


def _if(x):
    """MATLAB `if x`: nonempty and all elements nonzero."""
    if sp.issparse(x):
        x = x.toarray()
    a = np.asarray(x)
    return bool(a.size) and bool(np.all(a != 0))


def _any(x):
    if sp.issparse(x):
        return x.count_nonzero() > 0
    return bool(np.any(np.asarray(x) != 0))


def _f(x):
    return float(np.ravel(np.asarray(x, dtype=float))[0])


def _a(x):
    """MATLAB numeric value as a flat float array (column-major)."""
    if x is None:
        return np.zeros(0)
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x).ravel(order='F')


def _isreal(x):
    if sp.issparse(x):
        return not np.iscomplexobj(x.data) or not np.any(x.data.imag)
    return not np.iscomplexobj(x) or not np.any(np.asarray(x).imag)


def _nargin(f):
    try:
        ps = inspect.signature(f).parameters.values()
    except (TypeError, ValueError):
        return -1
    if any(p.kind == p.VAR_POSITIONAL for p in ps):
        return -1
    return sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in ps)


def _csc(x):
    return x.tocsc() if sp.issparse(x) else sp.csc_matrix(np.atleast_2d(np.asarray(x)))


def _solve(Q, b):
    """Q\\b. SuperLU is slow on the fairly dense general Liouvillians (1296 x 1296 with 8 % fill:
    2 s per solve), so moderately sized, not very sparse systems go to dense LAPACK LU (0.1 s)."""
    n = Q.shape[0]
    if n <= 4000 and Q.nnz > 0.002 * n * n:
        return np.linalg.solve(Q.toarray(), b.astype(complex))
    return spla.spsolve(Q.tocsc(), b)


def _processbasis(Basis, maxPotentialK, I, Symmetry):
    I = _a(I)
    nNuclei = I.size
    nobetatilts = Symmetry.nobetatilts
    tensorsCollinear = Symmetry.tensorsCollinear
    Basis.evenLmax = _f(Basis.LLMK[0])
    Basis.oddLmax = _f(Basis.LLMK[1])
    Basis.Mmax = _f(Basis.LLMK[2])
    Basis.Kmax = _f(Basis.LLMK[3])
    if isempty(Basis.jKmin):
        Basis.jKmin = -1.0
        if _if(tensorsCollinear):
            Basis.jKmin = +1.0
    if isempty(Basis.evenK):
        Basis.evenK = bool(_if(nobetatilts))
    if nNuclei == 0:
        Basis.pImax = 0.0
        Basis.pImaxall = 0.0
    else:
        if isempty(Basis.pImax):
            Basis.pImax = 2 * I
        Basis.pImax = np.minimum(_a(Basis.pImax), 2 * I)
        if isempty(Basis.pImaxall):
            Basis.pImaxall = float(np.sum(Basis.pImax))
        Basis.pImaxall = min(_f(Basis.pImaxall), float(np.sum(2 * I)))
        Basis.pImax = Basis.pImax.reshape(1, -1)
    if nNuclei == 2:
        Basis.pI1max = float(Basis.pImax[0, 0])
        Basis.pI2max = float(Basis.pImax[0, 1])
    deltaK = float(_if(Basis.evenK)) + 1
    Basis.v = np.concatenate([[Basis.evenLmax, Basis.oddLmax, Basis.Kmax, Basis.Mmax,
                               _f(Basis.jKmin), _f(Basis.pSmin), deltaK, float(_if(Basis.MpSymm))],
                              _a(Basis.pImax)]).reshape(1, -1)
    return Basis


def _processdynamics(Dyn, FieldSweep):
    err = ''
    if isfield(Dyn, 'Diff'):
        if np.any(_a(Dyn.Diff) < 0):
            raise ValueError('Sys.Diff cannot be negative.')
        Dyn.R = _a(Dyn.Diff)
    elif isfield(Dyn, 'logDiff'):
        Dyn.R = 10.0 ** _a(Dyn.logDiff)
    elif isfield(Dyn, 'tcorr'):
        Dyn.R = 1 / 6 / _a(Dyn.tcorr)
    elif isfield(Dyn, 'logtcorr'):
        if np.any(_a(Dyn.logtcorr) >= 0):
            raise ValueError('Sys.logtcorr must be negative.')
        Dyn.R = 1 / 6 / 10.0 ** _a(Dyn.logtcorr)
    else:
        err = ('You must specify a rotational correlation time or a diffusion tensor\n'
               '(Sys.tcorr, Sys.logtcorr, Sys.Diff or Sys.logDiff).')
        return Dyn, err
    R = Dyn.R
    if np.any(R < 0):
        raise ValueError('Negative diffusion rate or correlation times are not possible.')
    elif np.any(R > 1e12):
        print('Diffusion rate very fast. Simulation might not converge.')
    elif np.any(R < 1e3):
        print('Diffusion rate very slow. Simulation might not converge.')
    if R.size == 1:
        Dyn.R = R[[0, 0, 0]]
    elif R.size == 2:
        Dyn.R = R[[0, 0, 1]]
    elif R.size == 3:
        pass
    else:
        return Dyn, 'Sys.Diff must have 1, 2 or 3 elements (isotropic, axial, rhombic).'
    Dyn.R = Dyn.R.reshape(1, -1)
    if isfield(Dyn, 'lw'):
        lw = _a(Dyn.lw)
        if lw.size > 1:
            LorentzFWHM = lw[1] * 28 * 1e6 if FieldSweep else lw[1] * 1e6
        else:
            LorentzFWHM = 0
        Dyn.T2 = 1 / LorentzFWHM / np.pi if LorentzFWHM != 0 else np.inf
    return Dyn, err


@nout(-1)
def chili(Sys=None, Exp=None, Opt=None, nargout=None, plot=False):
    """chili(Sys, Exp, Opt) -> (x, spec); nargout=1 -> spec, nargout=3 -> (x, spec, info).
    Slow-motion cw EPR spectra from the stochastic Liouville equation (Lanczos, backslash,
    eigenvalue, conjugate-gradient or BiCGSTAB solvers; fast S=1/2 or general Liouvillian; MOMD
    grid for orientational potentials). plot=True shows the spectrum via cwepr_plot (MATLAB's
    nargout=0 behaviour). Opt.Diagnostics stores the diagnostics struct in chili.DIAGNOSTICS."""
    if nargout is None:
        nargout = 2
    if Sys is None or Exp is None:
        raise ValueError('Wrong number of input arguments!')
    if nargout < 0:
        raise ValueError('Not enough output arguments.')
    if nargout > 3:
        raise ValueError('Too many output arguments.')
    if isinstance(Sys, dict):
        Sys = Struct(Sys)
    Exp = Struct(Exp)
    Opt = Struct() if Opt is None or not isinstance(Opt, dict) else Struct(Opt)
    if not isfield(Opt, 'Verbosity'):
        Opt.Verbosity = 0
    config.LogLevel = Opt.Verbosity

    singleIso = isinstance(Sys, dict) and isfield(Sys, 'singleiso') and _if(Sys.singleiso)
    if not singleIso:
        F.logmsg(1, '-- slow motion regime simulation ----------------------------------')

    Exp.FrequencySweep = (not isfield(Exp, 'mwFreq')) and isfield(Exp, 'Field')
    if Exp.FrequencySweep:
        SweepAutoRange = (not isfield(Exp, 'mwRange') or isempty(Exp.mwRange)) and \
            (not isfield(Exp, 'mwCenterSweep') or isempty(Exp.mwCenterSweep))
    else:
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
    if separateTransitionSpectra:
        raise ValueError("chili does not support Opt.separate='transitions'.")
    if separateSiteSpectra:
        raise ValueError("chili does not support Opt.separate='sites'.")
    if separateOrientationSpectra:
        raise ValueError("chili does not support Opt.separate='orientations'.")

    singleIsotopologue = isinstance(Sys, dict) and isfield(Sys, 'singleiso') and _if(Sys.singleiso)
    if not singleIsotopologue:
        thirdOutput = nargout >= 3
        xAxis, spec, info = F.compisoloop(F.chili, Sys, Exp, Opt, SweepAutoRange, thirdOutput,
                                          separateComponentSpectra)
        if plot:
            F.cwepr_plot(xAxis, spec, Exp)
        if nargout == 0:
            return None
        if nargout == 1:
            return spec
        if nargout == 2:
            return xAxis, spec
        return xAxis, spec, info

    F.logmsg(1, '-- component spectrum simulation ----------------------------------')
    if not isfield(Sys, 'Nucs'):
        Sys.Nucs = ''
    Sys, err = F.validatespinsys(Sys)
    if not isempty(err):
        raise ValueError(err)
    Sys = Struct(Sys)
    if _if(Sys.MO_present):
        raise ValueError('chili does not support Sys.Ham* parameters.')
    if _any(Sys.L):
        raise ValueError('chili does not support Sys.L.')
    if isfield(Sys, 'nn') and _any(Sys.nn):
        raise ValueError('chili does not support nuclear-nuclear couplings (Sys.nn).')
    if _any(Sys.HStrain) or _any(Sys.gStrain) or _any(Sys.AStrain) or _any(Sys.DStrain):
        raise ValueError('chili does not support strains (HStrain, gStrain, AStrain, DStrain).')

    ConvolutionBroadening = bool(_a(Sys.lw)[0] > 0)
    if not isfield(Sys, 'DiffFrame'):
        Sys.DiffFrame = np.array([[0.0, 0.0, 0.0]])
    if not isfield(Sys, 'Potential'):
        Sys.Potential = np.zeros((0, 0))

    Dynamics = Struct()
    if isfield(Sys, 'lwpp'):
        Dynamics.lwpp = Sys.lwpp
    if isfield(Sys, 'lw'):
        Dynamics.lw = Sys.lw
    if not isfield(Sys, 'Exchange'):
        Sys.Exchange = 0
    Dynamics.Exchange = Sys.Exchange
    Dynamics.Exchange = _f(Dynamics.Exchange) * 2 * np.pi * 1e6

    DiffFields = ['logtcorr', 'tcorr', 'Diff', 'logDiff']
    hasDiffFields = [isfield(Sys, f) for f in DiffFields]
    if sum(hasDiffFields) > 1:
        raise ValueError('Only one of Sys.tcorr, Sys.logtcorr, Sys.Diff, and Sys.logDiff is allowed.')
    elif not any(hasDiffFields):
        raise ValueError('One of Sys.tcorr, Sys.logtcorr, Sys.Diff, and Sys.logDiff is required.')
    for k, fld in enumerate(DiffFields):
        if hasDiffFields[k]:
            Dynamics[fld] = Sys[fld]

    if isfield(Sys, 'lambda') and not isempty(Sys['lambda']):
        lam = _a(Sys['lambda'])
        if lam.size > 4:
            raise ValueError('Sys.lambda must be a vector with at most 4 elements.')
        LMK = [[2, 0, 0], [2, 0, 2], [4, 0, 0], [4, 0, 2]]
        s = '    Sys.Potential = [' + '; '.join('%d %d %d %g' % (*LMK[p], lam[p]) for p in range(lam.size)) + ']'
        raise ValueError('\n  Sys.lambda is obsolete.\n  Use the following instead:\n\n%s;    %% L M K lambda\n' % s)

    Potential = Struct()
    if not isempty(Sys.Potential):
        Pot = np.atleast_2d(np.asarray(Sys.Potential))
        if Pot.shape[1] != 4:
            raise ValueError('Sys.Potential needs 4 entries per row (L, M, K, lambda).')
        Potential.L = Pot[:, 0].real.reshape(-1, 1)
        Potential.M = Pot[:, 1].real.reshape(-1, 1)
        Potential.K = Pot[:, 2].real.reshape(-1, 1)
        lamv = Pot[:, 3]
        Potential['lambda'] = (lamv if np.any(np.imag(lamv)) else np.real(lamv)).reshape(-1, 1)
        rmv = vec(Potential['lambda']) == 0
        rmv = rmv | ((vec(Potential.L) == 0) & (vec(Potential.M) == 0) & (vec(Potential.K) == 0))
        if np.any(rmv):
            for f in ('L', 'M', 'K', 'lambda'):
                Potential[f] = Potential[f][~rmv]
                Potential[f] = Potential[f].reshape(-1, 1)
    else:
        for f in ('L', 'M', 'K', 'lambda'):
            Potential[f] = np.zeros((0, 0))
    usePotential = not isempty(Potential['lambda'])
    oldStylePotential = False
    if usePotential:
        PL, PM, PK, Plam = vec(Potential.L), vec(Potential.M), vec(Potential.K), vec(Potential['lambda'])
        if np.any(PL < 0):
            raise ValueError('L values of potential coefficients must be nonnegative.')
        if np.any(np.abs(PK) > PL):
            raise ValueError('L and K values of potential coefficients do not satisfy |K|<=L.')
        if np.any(np.abs(PM) > PL):
            raise ValueError('L and M values of potential coefficients do not satisfy |M|<=L.')
        if np.any(PK < 0):
            raise ValueError('Only potential terms with nonnegative values of K are allowed. Terms with negative K '
                             'required to render the potential real-valued are supplemented automatically.')
        if np.any(PM[PK == 0] < 0):
            raise ValueError('For potential terms with K=0, M must be nonnegative. Terms with negative M required '
                             'to render the potential real-valued are supplemented automatically.')
        zeroMK = (PK == 0) & (PM == 0)
        if np.iscomplexobj(Plam) and np.any(Plam[zeroMK].imag != 0):
            raise ValueError('Potential coefficients for M=K=0 must be real-valued.')
        oldStylePotential = (bool(np.all((PL == 2) | (PL == 4))) and bool(np.all(PM == 0)) and
                             bool(np.all((PK == 0) | (PK == 2))) and _isreal(Plam))

    # Experimental parameters
    if not isfield(Exp, 'nPoints'):
        Exp.nPoints = 1024
    if not isfield(Exp, 'Harmonic'):
        Exp.Harmonic = np.zeros((0, 0))
    if not isfield(Exp, 'mwPhase'):
        Exp.mwPhase = 0
    if not isfield(Exp, 'Temperature'):
        Exp.Temperature = np.nan
    if not isfield(Exp, 'ModAmp'):
        Exp.ModAmp = 0
    if not isfield(Exp, 'mwMode'):
        Exp.mwMode = 'perpendicular'
    if not isfield(Exp, 'Ordering'):
        Exp.Ordering = np.zeros((0, 0))
    if not isfield(Exp, 'SampleFrame'):
        Exp.SampleFrame = np.zeros((0, 0))
    if isfield(Exp, 'lightBeam') and not isempty(Exp.lightBeam):
        raise ValueError('Photoselection (via Exp.lightBeam) is not supported.')
    nP = np.asarray(Exp.nPoints)
    if np.iscomplexobj(nP) or nP.size > 1 or _f(nP) < 2:
        raise ValueError('Problem with Exp.nPoints. Needs to be a number >= 2.')
    nPoints = int(_f(nP))
    if not np.all(np.isnan(_a(Exp.Temperature))):
        T_ = _a(Exp.Temperature)
        if T_.size != 1 or np.isinf(T_[0]) or T_[0] < 0:
            raise ValueError('Problem with Exp.Temperature. If given, Exp.Temperature must be a positive value.')

    F.logmsg(1, 'Experiment:')
    if not isfield(Exp, 'mwFreq'):
        if not isfield(Exp, 'Field'):
            raise ValueError('Please supply either the microwave frequency in Exp.mwFreq (for field sweeps) or the '
                             'magnetic field in Exp.Field (for frequency sweeps).')
        FieldSweep = False
    else:
        if isfield(Exp, 'Field'):
            raise ValueError('Give either Exp.mwFreq (for a field sweep) or Exp.Field (for a frequency sweep), '
                             'but not both.')
        FieldSweep = True

    if FieldSweep:
        mw = np.asarray(Exp.mwFreq)
        if mw.size != 1 or np.any(mw <= 0) or np.iscomplexobj(mw):
            raise ValueError('Uninterpretable microwave frequency in Exp.mwFreq.')
        Exp.mwFreq = _f(mw)
        F.logmsg(1, '  field sweep, mw frequency %0.8g GHz', Exp.mwFreq)
    else:
        Bf = np.asarray(Exp.Field)
        if Bf.size != 1 or np.any(Bf <= 0) or np.iscomplexobj(Bf):
            raise ValueError('Uninterpretable magnetic field in Exp.Field.')
        Exp.Field = _f(Bf)
        F.logmsg(1, '  frequency sweep, magnetic field %0.8g T', Exp.Field)

    if FieldSweep:
        if isfield(Exp, 'CenterSweep'):
            if isfield(Exp, 'Range'):
                F.logmsg(0, 'Using Exp.CenterSweep and ignoring Exp.Range.')
        else:
            if isfield(Exp, 'Range'):
                Rg = _a(Exp.Range)
                Exp.CenterSweep = np.array([(Rg[0] + Rg[1]) / 2, Rg[1] - Rg[0]])
            else:
                if int(_f(Sys.nElectrons)) == 1 and _f(Sys.S) == 1 / 2:
                    F.logmsg(1, '  automatic determination of sweep range')
                    Stretch = 1.25
                    I = _a(F.nucspin(Sys.Nucs))
                    if I.size > 0:
                        Amax = np.max(np.abs(np.atleast_2d(np.asarray(Sys.A, dtype=float))), axis=1)
                        if I.size != 1 and I.size != Amax.size:
                            raise ValueError('Arrays have incompatible sizes for this operation.')
                        hf = float(np.sum(I * Amax)) * 1e6
                    else:
                        hf = 0
                    gmax = float(np.max(np.asarray(Sys.g, dtype=float)))
                    gmin = float(np.min(np.asarray(Sys.g, dtype=float)))
                    planck, bmagn = _c('planck'), _c('bmagn')
                    minB = planck * (Exp.mwFreq * 1e9 - hf) / bmagn / gmax / 1e-3
                    maxB = planck * (Exp.mwFreq * 1e9 + hf) / bmagn / gmin / 1e-3
                    Exp.CenterSweep = np.array([(maxB + minB) / 2, Stretch * max(maxB - minB, 5)])
                else:
                    raise ValueError('Cannot automatically determine sweep range for this spin system.')
    else:
        if isfield(Exp, 'mwCenterSweep'):
            if isfield(Exp, 'mwRange'):
                F.logmsg(0, 'Using Exp.mwCenterSweep and ignoring Exp.mwRange.')
        else:
            if isfield(Exp, 'mwRange'):
                Rg = _a(Exp.mwRange)
                Exp.mwCenterSweep = np.array([(Rg[0] + Rg[1]) / 2, Rg[1] - Rg[0]])
            else:
                raise ValueError('Either Exp.mwRange or Exp.mwCenterSweep need to be given.')

    if FieldSweep:
        cs = _a(Exp.CenterSweep)
        CenterField = cs[0]
        SweepWidth = cs[1]
        Exp.Range = cs[0] + np.array([-0.5, 0.5]) * SweepWidth
        if np.any(Exp.Range < 0) or Exp.Range[1] - Exp.Range[0] <= 0:
            raise ValueError('Invalid sweep range! Check Exp.CenterSweep or Exp.Range.')
        F.logmsg(1, '  field range (mT): min %g, max %g, center %g, width %g',
                 Exp.Range[0], Exp.Range[1], CenterField, SweepWidth)
    else:
        cs = _a(Exp.mwCenterSweep)
        CenterFreq = cs[0]
        SweepWidth = cs[1]
        Exp.mwRange = cs[0] + np.array([-0.5, 0.5]) * SweepWidth
        CenterField = Exp.Field
        if np.any(Exp.mwRange < 0) or Exp.mwRange[1] - Exp.mwRange[0] <= 0:
            raise ValueError('Invalid sweep range! Check Exp.mwCenterSweep or Exp.mwRange.')
        F.logmsg(1, '  frequency range (GHz): min %g, max %g, center %g, width %g',
                 Exp.mwRange[0], Exp.mwRange[1], CenterFreq, SweepWidth)

    if isempty(Exp.Harmonic) or np.any(np.isnan(_a(Exp.Harmonic))):
        Exp.Harmonic = 1 if FieldSweep else 0
    if not np.any(_f(Exp.Harmonic) == np.array([-1, 0, 1, 2])):
        raise ValueError('Exp.Harmonic must be 0, 1 or 2.')
    Harmonic = int(_f(Exp.Harmonic))
    ma = _a(Exp.ModAmp)
    if np.any(ma < 0) or np.any(np.isnan(ma)) or ma.size != 1:
        raise ValueError('Exp.ModAmp must be either a single positive number or zero.')
    ModAmp = float(ma[0])
    if ModAmp > 0:
        if FieldSweep:
            F.logmsg(1, '  field modulation, amplitude %g mT', ModAmp)
            if Harmonic < 1:
                raise ValueError('With field modulation (Exp.ModAmp), Exp.Harmonic=0 does not work.')
            Exp.ModHarmonic = Harmonic
            Exp.ConvHarmonic = 0
            Exp.DerivHarmonic = 0
        else:
            raise ValueError('Exp.ModAmp cannot be used with frequency sweeps.')
    else:
        Exp.ModHarmonic = 0
        if ConvolutionBroadening:
            Exp.ConvHarmonic = Harmonic
            Exp.DerivHarmonic = 0
        else:
            Exp.ConvHarmonic = 0
            Exp.DerivHarmonic = Harmonic

    if Exp.mwMode == 'perpendicular':
        ParallelMode = False
    elif Exp.mwMode == 'parallel':
        ParallelMode = True
    else:
        raise ValueError("Exp.mwMode must be either 'perpendicular' or 'parallel'.")
    F.logmsg(1, '  harmonic %d, %s mode', Harmonic, Exp.mwMode)
    if ParallelMode:
        raise ValueError('chili does not support parallel-mode spectra.')
    F.logmsg(1, '  %d points', nPoints)
    if isfield(Exp, 'CrystalSymmetry'):
        import warnings
        warnings.warn('Exp.CrystalSymmetry is not used by chili.')

    if callable(Exp.Ordering) or not isempty(Exp.Ordering):
        if not callable(Exp.Ordering) and np.size(Exp.Ordering) == 1 and np.isrealobj(Exp.Ordering):
            lambda_ = _f(Exp.Ordering)
            Exp.Ordering = lambda alpha, beta, gamma: np.exp(lambda_ * F.plegendre(2, 0, np.cos(beta)))
            F.logmsg(1, '  partial director ordering (built-in function, coefficient = %g)', lambda_)
        elif callable(Exp.Ordering):
            F.logmsg(1, '  partial director ordering (user-supplied function)')
        else:
            raise ValueError('Exp.Ordering must be either a single number or a function handle.')
        if _nargin(Exp.Ordering) == 1:
            f1 = Exp.Ordering
            Exp.Ordering = lambda alpha, beta, gamma: f1(beta) * np.ones(np.shape(gamma))
        elif _nargin(Exp.Ordering) != 3:
            F.logmsg(1, '  Ordering function in Exp.Ordering must take 3 input arguments (alpha,beta,gamma).')
    else:
        F.logmsg(1, '  director ordering: none')
    partiallyOrderedSample = callable(Exp.Ordering) or not isempty(Exp.Ordering)
    integrateOverGrid = isempty(Exp.SampleFrame) and usePotential

    # Options
    if not isfield(Opt, 'LLMK'):
        if usePotential:
            raise ValueError('Sys.Potential is given. Please provide basis set information (Opt.LLMK etc).')
        Opt.LLMK = np.array([[14.0, 7.0, 2.0, 6.0]])
    defaults = [('highField', False), ('pImax', np.zeros((0, 0))), ('pImaxall', np.zeros((0, 0))),
                ('GridSize', np.array([[19.0, 0.0]])), ('LiouvMethod', ''), ('FieldSweepMethod', np.zeros((0, 0))),
                ('PostConvNucs', ''), ('Solver', ''), ('GridSymmetry', 'Dinfh')]
    for k_, v_ in defaults:
        if not isfield(Opt, k_):
            Opt[k_] = v_
    if isfield(Opt, 'LLKM'):
        raise ValueError('Opt.LLKM is not used by chili(), provide Opt.LLMK instead.')
    if isfield(Opt, 'nKnots'):
        raise ValueError('Options.nKnots is obsolete. Use Options.GridSize instead, e.g. Options.GridSize = 91.')
    if isfield(Opt, 'Symmetry'):
        raise ValueError("Options.Symmetry is obsolete. Use Options.GridSymmetry instead, e.g. "
                         "Options.GridSymmetry = 'D2h'.")
    if isfield(Opt, 'Output'):
        raise ValueError('Options.Output is obsolete. Use Opt.separate instead.')
    defaults = [('Rescale', True), ('Threshold', 1e-6), ('Lentz', True), ('IncludeNZI', True), ('pqOrder', False),
                ('GridFrame', np.zeros((0, 0))), ('Diagnostics', ''), ('useLMKbasis', False),
                ('useStartvecSelectionRules', True), ('PeqTol', np.zeros((0, 0))), ('jKmin', np.zeros((0, 0))),
                ('evenK', np.zeros((0, 0)))]
    for k_, v_ in defaults:
        if not isfield(Opt, k_):
            Opt[k_] = v_
    if isfield(Opt, 'deltaK'):
        raise ValueError('Opt.deltaK is obsolete. Use Opt.evenK instead.')
    if not isinstance(Opt.Diagnostics, str) and not isempty(Opt.Diagnostics):
        raise ValueError('If given, Opt.Diagnosics must be a valid Matlab variable name.')
    saveDiagnostics = not isempty(Opt.Diagnostics)

    nElectrons = int(_f(Sys.nElectrons))
    nNuclei = int(_f(Sys.nNuclei))
    Sval = _a(Sys.S)
    if not isfield(Opt, 'LiouvMethod') or isempty(Opt.LiouvMethod):
        if nElectrons == 1 and Sval[0] == 1 / 2 and nNuclei <= 2 and (not usePotential or oldStylePotential):
            Opt.LiouvMethod = 'fast'
        else:
            Opt.LiouvMethod = 'general'
    LiouvMethod, err = F.parseoption(Opt, 'LiouvMethod', ['fast', 'general'], nargout=2)
    if not isempty(err):
        raise ValueError(err)
    generalLiouvillian = LiouvMethod == 2
    if generalLiouvillian:
        if _any(Sys.Exchange):
            raise ValueError("Opt.LiouvMethod='general' does not support spin exchange (Sys.Exchange).")
    else:
        if nElectrons > 1:
            raise ValueError("Opt.LiouvMethod='fast' does not work with more than one electron spin.")
        if Sval[0] != 1 / 2:
            raise ValueError("Opt.LiouvMethod='fast' does not work for S>1/2.")
        if nNuclei > 2:
            raise ValueError("Opt.LiouvMethod='fast' does not work more than 2 nuclei in the spin system.")
        if usePotential and not oldStylePotential:
            raise ValueError("Opt.LiouvMethod='fast' does not work with this orientational potential.")

    doPostConvolution = not isempty(Opt.PostConvNucs)
    if doPostConvolution:
        pc = np.unique(_a(Opt.PostConvNucs))
        Opt.PostConvNucs = pc.reshape(1, -1)
        if np.any(pc < 1) or np.any(pc > nNuclei):
            raise ValueError('Opt.PostConvNucs must contain indices of nuclei (1 to %d).' % nNuclei)
        nPostConvNucs = pc.size
        if nNuclei - nPostConvNucs > 2 and not generalLiouvillian:
            raise ValueError('Cannot have more than two nuclei for the Stochastic Liouville equation with this '
                             'Opt.LiouvMethod.')
        fullSys = Sys
        Sys = Struct(F.nucspinrmv(Sys, Opt.PostConvNucs))
        Sys.processed = False
        Sys, err = F.validatespinsys(Sys)
        if not isempty(err):
            raise ValueError(err)
        Sys = Struct(Sys)
        nElectrons = int(_f(Sys.nElectrons))
        nNuclei = int(_f(Sys.nNuclei))

    if np.any(_a(Sys.n) != 1):
        raise ValueError('chili cannot handle systems with nuclei with Sys.n > 1 only if these nuclei are treated '
                         'using post-convolution (Opt.PostConvNucs).')
    GridSize = list(_a(Opt.GridSize))
    if len(GridSize) < 1:
        GridSize.append(19.0)
    if len(GridSize) < 2:
        GridSize.append(0.0)
    Opt.GridSize = np.array([GridSize])
    if GridSize[1] != 0:
        raise ValueError('chili cannot interpolate orientations. Set Opt.GridSize(2) to zero.')
    LLMK = _a(Opt.LLMK)
    if (np.iscomplexobj(LLMK) or LLMK.size != 4 or np.any(LLMK < 0) or np.any(np.mod(LLMK, 1))):
        raise ValueError('Opt.LLMK must be a 4-element array with non-negative integers.')
    if LLMK[0] % 2:
        raise ValueError('Opt.LLMK(1) must be an even number.')
    if LLMK[1] > 0 and not LLMK[1] % 2:
        raise ValueError('Opt.LLMK(2) must be an odd integer.')
    maxL = max(LLMK[0], LLMK[1])
    if LLMK[2] > maxL:
        raise ValueError('Opt.LLMK(3)=%d, but must be smaller or equal to the maximum L (%d).' % (LLMK[2], maxL))
    if LLMK[3] > maxL:
        raise ValueError('Opt.LLMK(4)=%d, but must be smaller or equal to the maximum L (%d).' % (LLMK[3], maxL))
    Basis = Struct()
    Basis.LLMK = LLMK
    Basis.jKmin = Opt.jKmin
    Basis.evenK = Opt.evenK
    if isfield(Opt, 'pSmin'):
        raise ValueError('Opt.pSmin is not supported - use Opt.highField instead.')
    Basis.pSmin = +1.0 if _if(Opt.highField) else -1.0
    if not isempty(Opt.pImax):
        pim = _a(Opt.pImax)
        if pim.size != nNuclei and pim.size != 1:
            raise ValueError('Opt.pImax must contain one entry for every nucleus or just a single number.')
        if np.any(pim < 0):
            raise ValueError('Every element in Opt.pImax must be 0 or larger.')
    Basis.pImax = Opt.pImax
    if not isempty(Opt.pImaxall):
        if np.size(Opt.pImaxall) != 1:
            raise ValueError('Opt.pImaxall must be a single non-negative number.')
    Basis.pImaxall = Opt.pImaxall
    if not isfield(Opt, 'MpSymm'):
        Opt.MpSymm = False
    Basis.MpSymm = Opt.MpSymm

    F.logmsg(1, 'Methods:')
    if FieldSweep:
        if isempty(Opt.FieldSweepMethod):
            Opt.FieldSweepMethod = 'approxlin'
        _, err = F.parseoption(Opt, 'FieldSweepMethod', ['explicit', 'approxinv', 'approxlin'], nargout=2)
        if not isempty(err):
            raise ValueError(err)
    fsm = Opt.FieldSweepMethod if isinstance(Opt.FieldSweepMethod, str) else ''
    explicitFieldSweep = FieldSweep and fsm == 'explicit'
    F.logmsg(1, '  field sweep method: %s', fsm)
    if isempty(Opt.Solver):
        Opt.Solver = '\\' if fsm == 'explicit' else 'L'
    useLanczosSolver = Opt.Solver == 'L'
    if explicitFieldSweep and Opt.Solver != '\\':
        raise ValueError("For an explicit field sweep, use Opt.Solver='\\'.")
    if Opt.Solver == 'L':
        if _f(Opt.Lentz) == 1:
            SolverString = 'Lanczos tridiagonalization, left-to-right continued fraction evaluation'
        else:
            SolverString = 'Lanczos tridiagonalization, right-to-left continued fraction evaluation'
    elif Opt.Solver == '\\':
        SolverString = 'backslash'
    elif Opt.Solver == 'E':
        SolverString = 'eigenvalue method, sum of Lorentzians'
    elif Opt.Solver == 'C':
        SolverString = 'conjugate gradients tridiagonalization, right-to-left continued fraction evaluation'
    elif Opt.Solver == 'B':
        SolverString = 'biconjugate gradients, stabilized'
    else:
        raise ValueError("Unknown method in Options.Solver. Must be 'L', '\\', 'E', 'C', or 'B'.")
    F.logmsg(1, '  solver: %s = %s', Opt.Solver, SolverString)

    if not generalLiouvillian:
        blockSize = 1e6
        minBlockSize = 1e3
        if not isfield(Opt, 'AllocationBlockSize'):
            Opt.AllocationBlockSize = blockSize
        Opt.AllocationBlockSize = _f(Opt.AllocationBlockSize)
        if Opt.AllocationBlockSize < minBlockSize:
            raise ValueError('Opt.AllocationBlockSize = %d is too small. Increase its value to at least %d.'
                             % (Opt.AllocationBlockSize, minBlockSize))
        F.logmsg(2, '  allocating memory in blocks of %d non-zero elements', Opt.AllocationBlockSize)

    Spins = _a(Sys.Spins)
    if generalLiouvillian:
        F.logmsg(1, '  construction of Liouvillian: general code')
        nSpins = Spins.size
        SpinOps = [None] * (3 * nSpins)
        for iSpin in range(nSpins, 0, -1):
            for k in range(3):
                SpinOps[(iSpin - 1) + k * nSpins] = F.sop(Sys.Spins, np.array([[iSpin, k + 1.0]]), 'sparse')
        SdetOp = None
        for e in range(nElectrons):
            SdetOp = SpinOps[e] if SdetOp is None else SdetOp + SpinOps[e]
        SdetOp = sp.csc_matrix(SdetOp)
    else:
        F.logmsg(1, '  Liouvillian construction: fast S=1/2 code')
        SpinOps = np.zeros((0, 0))

    includeNQI = False
    T, F_, Sys, Symmetry, isFieldDep = F.magint(Sys, SpinOps, CenterField, Opt.IncludeNZI, includeNQI,
                                                explicitFieldSweep)
    Sys = Struct(Sys)
    diagnostics = Struct()
    if saveDiagnostics:
        diagnostics.T = T
        diagnostics.F = F_
    noAnisotropiesPresent = not _any(F_['F1']) and not _any(F_['F2'])
    if noAnisotropiesPresent:
        raise ValueError('This is an isotropic spin system. chili cannot calculate a slow-motion spectrum.')

    Dynamics, err = _processdynamics(Dynamics, FieldSweep)
    if err:
        raise ValueError(err)
    maxPotK = float(np.max(_a(Potential.K))) if np.size(Potential.K) else np.zeros((0, 0))
    Basis = _processbasis(Basis, maxPotK, Sys.I, Symmetry)

    if _if(Sys.fullg):
        gfull = np.asarray(Sys.g)
        mean_g = []
        for iElectron in range(nElectrons):
            ev = np.linalg.eigvals(gfull[3 * iElectron:3 * iElectron + 3, :])
            mean_g.append(np.mean(ev))
        gavg = np.mean(mean_g)
        gavg = gavg.real if np.iscomplexobj(gavg) and gavg.imag == 0 else gavg
    else:
        g2 = np.atleast_2d(np.asarray(Sys.g, dtype=float))
        gavg = float(np.mean(np.mean(g2, axis=0))) if g2.shape[0] > 1 else float(np.sum(g2) / g2.size)

    if FieldSweep:
        B_ = _linspace(Exp.Range[0], Exp.Range[1], nPoints)
        if fsm == 'explicit':
            B0 = B_
            nu = np.array([Exp.mwFreq])
        elif fsm == 'approxinv':
            B0 = np.array([CenterField])
            nu = Exp.mwFreq * B0 / B_
        elif fsm == 'approxlin':
            B0 = np.array([CenterField])
            nu = Exp.mwFreq - _a(F.unitconvert((B_ - B0).reshape(1, -1), 'mT->MHz', gavg)) / 1e3
        else:
            raise ValueError("Unknown setting '%s' in Opt.FieldSweepMethod." % fsm)
    else:
        B0 = np.array([CenterField])
        nu = _linspace(Exp.mwRange[0], Exp.mwRange[1], nPoints)
    B0 = B0 / 1e3
    T2 = Dynamics.T2
    omega0 = (2 * np.pi * nu * 1e9) + 1j * (1 / T2)
    if FieldSweep:
        xAxis = _linspace(Exp.Range[0], Exp.Range[1], nPoints)
    else:
        xAxis = nu

    if integrateOverGrid:
        if GridSize[0] == 1:
            phi = np.array([0.0])
            theta = np.array([0.0])
            gridWeights = np.array([4 * np.pi])
            Opt.GridSymmetry = 'O3'
        else:
            grid = F.sphgrid(Opt.GridSymmetry, GridSize[0])
            Vecs = np.asarray(grid.vecs)
            gridWeights = _a(grid.weights)
            if isempty(Opt.GridFrame):
                phi, theta = F.vec2ang(Vecs)
            else:
                phi, theta = F.vec2ang(np.asarray(Opt.GridFrame) @ Vecs)
            phi, theta = _a(phi), _a(theta)
        F.logmsg(1, '  grid simulation with %d orientations, grid symmetry %s (director frame relative to lab '
                 'frame)', phi.size, Opt.GridSymmetry)
    else:
        if not isempty(Exp.SampleFrame):
            sf = _a(Exp.SampleFrame)
            phi = np.array([-sf[2]])
            theta = np.array([-sf[1]])
        else:
            phi = np.array([0.0])
            theta = np.array([0.0])
        gridWeights = np.array([4 * np.pi])
        F.logmsg(2, '  single-orientation simulation')
    nOrientations = phi.size
    Basis.DirTilt = bool(np.any(theta != 0))

    if partiallyOrderedSample:
        orderingWeights = _a(Exp.Ordering(0, -theta.reshape(1, -1), -phi.reshape(1, -1)))
        if np.any(orderingWeights < 0):
            raise ValueError('User-supplied orientation distribution gives negative values!')
        if np.all(orderingWeights == 0):
            raise ValueError('User-supplied orientation distribution is all-zero.')
        F.logmsg(2, '  orientational potential')
    else:
        orderingWeights = np.ones(nOrientations)
    Weights = gridWeights * orderingWeights
    Weights = 4 * np.pi * Weights / np.sum(Weights)

    F.logmsg(1, 'Basis set:')
    F.logmsg(1, '  truncation settings:')
    F.logmsg(1, '    orientational: Leven max %d, Lodd max %d, Mmax %d, Kmax %d, evenK %d, jKmin %+d',
             Basis.LLMK[0], Basis.LLMK[1], Basis.LLMK[2], Basis.LLMK[3], int(_if(Basis.evenK)), _f(Basis.jKmin))
    F.logmsg(1, '    spin: pSmin %+d, pImax %s, pImaxall %d', Basis.pSmin,
             ' '.join('%g' % x for x in _a(Basis.pImax)), _f(Basis.pImaxall))
    F.logmsg(1, '    M-p symmetry: %d', int(_if(Basis.MpSymm)))

    nStates = int(_f(Sys.nStates))
    keep = None
    idxpq = None
    if generalLiouvillian:
        Basis = Struct(F.generateoribasis(Basis, 'LMK' if _if(Opt.useLMKbasis) else 'LjKKM'))
        nOriBasis = np.size(Basis.L)
        nSpinBasis = nStates ** 2
        F.logmsg(1, '  orientational basis: %d functions', nOriBasis)
        pqo = F.pqorder(Sys.Spins)
        idxpq = _a(pqo[0]).astype(int) - 1
        pq = np.atleast_2d(np.asarray(pqo[1], dtype=float))
        SdetOp = SdetOp.reshape((nSpinBasis, 1), order='F').tocsc()
        if _if(Opt.pqOrder):
            SdetOp = SdetOp[idxpq, :]
        keep = np.ones(nSpinBasis, dtype=bool)
        for ie in range(nElectrons):
            keep = keep & (pq[:, 2 * ie] >= Basis.pSmin)
        pIsum = 0
        pImaxv = _a(Basis.pImax)
        for in_ in range(nNuclei):
            pI = pq[:, 2 * nElectrons + 2 * in_]
            pIsum = pIsum + pI
            keep = keep & (np.abs(pI) <= pImaxv[in_])
        keep = keep & (np.abs(pIsum) <= _f(Basis.pImaxall))
        if _if(Opt.pqOrder):
            keep = keep[idxpq]
        F.logmsg(1, '  spin basis: %d functions (%0.1f%% of %d)', int(np.sum(keep)),
                 np.sum(keep) / nSpinBasis * 100, nSpinBasis)
        keep = np.tile(keep, nOriBasis)
        if _if(Opt.MpSymm):
            M = _a(Basis.M)
            psum = np.sum(pq[:, 0::2], axis=1)
            keep_Mp = (psum.reshape(-1, 1) - M.reshape(1, -1)) == 1
            keep = keep & keep_Mp.ravel(order='F')
            F.logmsg(1, '  applying M-p symmetry: keeping %d of %d functions', int(np.sum(keep)), keep.size)
        F.logmsg(1, '  total basis size: %d functions', int(np.sum(keep)))
    else:
        Basis = Struct(F.chili_basisbuild(Basis, Sys))
        F.logmsg(1, '  total basis size: %d functions', np.size(Basis.L))
    if saveDiagnostics:
        diagnostics.basis = Basis

    if generalLiouvillian:
        F.logmsg(1, 'Precalculating 3j symbols up to Leven=%d and Lodd=%d...', Basis.LLMK[0], Basis.LLMK[1])
        computeRankOne = _any(F_['F1'])
        jjj0, jjj1, jjj2 = F.jjjsymbol(Basis.LLMK[0], Basis.LLMK[1], computeRankOne)

    if usePotential:
        F.logmsg(1, 'Calculating Wigner expansion coefficients for potential-dependent part of diffusion matrix')
        XLMK = F.chili_xlmk(Potential, Dynamics.R)
    else:
        XLMK = []

    if generalLiouvillian:
        F.logmsg(1, 'Calculating the diffusion matrix')
        Gamma = F.diffsuperop(Basis, Dynamics.R, XLMK, Potential)
        Gamma = _csc(F.kroneye(Gamma, nStates ** 2))
        Gamma = Gamma[keep, :][:, keep]
        if config.LogLevel is not None and config.LogLevel >= 1:
            ga = np.abs(Gamma.data) if Gamma.nnz else np.zeros(1)
            F.logmsg(1, '  imag/real = %f', float(np.max(np.abs(Gamma.data.imag), initial=0) /
                                                np.max(np.abs(Gamma.data.real), initial=0)) if ga.size else 0)

    F.logmsg(1, 'Calculating starting vector...')
    nInt = []
    if generalLiouvillian:
        if not isfield(Opt, 'StartVec') or isempty(Opt.StartVec):
            Opt_ = Struct(useSelectionRules=Opt.useStartvecSelectionRules, PeqTolerances=Opt.PeqTol)
            sqrtPeq, nInt = F.chili_eqpopvec(Basis, Potential, Opt_)
            sqrtPeq = _csc(sqrtPeq)
            if sqrtPeq.shape[1] != 1:
                sqrtPeq = sqrtPeq.reshape((-1, 1), order='F').tocsc()
            sd = SdetOp / spla.norm(SdetOp)
            StartVector = sp.kron(sqrtPeq, sd, format='csc')
            StartVector = StartVector[keep, :]
            normPeqVec = spla.norm(sqrtPeq) ** 2
            F.logmsg(1, '  norm of Peq vector: %g', normPeqVec)
            if normPeqVec < 0.99:
                print('The norm of the equilibrium population vector in this truncated basis is %0.3g.\n'
                      ' It should be close to 1. The basis might be too small.' % normPeqVec)
        else:
            F.logmsg(1, '  using provided vector')
            if np.size(Opt.StartVec) == int(np.sum(keep)):
                StartVector = _csc(np.asarray(Opt.StartVec).reshape(-1, 1, order='F'))
                nInt = []
            else:
                raise ValueError('Opt.StartVec must have %d elements.' % int(np.sum(keep)))
        StartVector = StartVector / spla.norm(StartVector)
    else:
        if usePotential:
            LMK = np.array([[2, 0, 0], [2, 0, 2], [4, 0, 0], [4, 0, 2]], dtype=float)
            PotLMK = np.column_stack([vec(Potential.L), vec(Potential.M), vec(Potential.K)])
            lambda_shortlist = np.zeros((4, 1), dtype=np.asarray(Potential['lambda']).dtype)
            for p in range(4):
                hit = np.nonzero(np.all(PotLMK == LMK[p], axis=1))[0]
                if hit.size:
                    lambda_shortlist[p, 0] = vec(Potential['lambda'])[hit[0]]
        else:
            lambda_shortlist = np.zeros((0, 0))
        StartVector = _csc(F.chili_startingvector(Basis, lambda_shortlist))
        nInt = []
    if saveDiagnostics:
        diagnostics.sv = StartVector

    BasisSize = StartVector.shape[0]
    F.logmsg(1, '  vector size: %dx1', BasisSize)
    F.logmsg(1, '  non-zero elements: %d/%d (%0.2f%%)', StartVector.nnz, BasisSize,
             100 * StartVector.nnz / BasisSize)
    F.logmsg(1, '  maxabs %g, norm %g', float(np.max(np.abs(StartVector.data), initial=0)),
             spla.norm(StartVector))
    if not isempty(nInt):
        nI = _a(nInt)
        F.logmsg(1, '  evaluated integrals: 1D %d, 2D %d, 3D %d', nI[0], nI[1], nI[2])

    if _if(Opt.useLMKbasis) and useLanczosSolver:
        TT = _csc(F.ksymmetrizer(Basis))
        TT = sp.kron(TT, sp.identity(nStates ** 2), format='csc')
        TT = TT[keep, :][:, keep]

    # director azimuth used in the Liouvillian (see EASYSPIN_DIRECTOR_PHI)
    phiL = phi if EASYSPIN_DIRECTOR_PHI else np.zeros_like(phi)
    spec = 0
    thisspec = None
    scale = 1
    L = None
    omega = omega0
    for iOri in range(nOrientations):
        F.logmsg(1, 'Orientation %d/%d: phi = %gdeg, theta = %gdeg (weight %g)', iOri + 1, nOrientations,
                 phi[iOri] * 180 / np.pi, theta[iOri] * 180 / np.pi, Weights[iOri])
        F.logmsg(1, 'Computing Liouvillian matrix...')
        if generalLiouvillian:
            Q0B, Q1B, Q2B, Q0G, Q1G, Q2G = F.rbos(T, F_, np.array([[phiL[iOri], theta[iOri], 0.0]]), isFieldDep)
            if _if(Opt.pqOrder):
                def perm(m):
                    return _csc(m)[idxpq, :][:, idxpq]
                Q0B = perm(Q0B)
                Q0G = perm(Q0G)
                Q1B = [perm(m) for m in Q1B] if isinstance(Q1B, list) else Q1B
                Q1G = [perm(m) for m in Q1G] if isinstance(Q1G, list) else Q1G
                Q2B = [perm(m) for m in Q2B]
                Q2G = [perm(m) for m in Q2G]
            LH = F.liouvhamiltonian_LMK if _if(Opt.useLMKbasis) else F.liouvhamiltonian
            if explicitFieldSweep:
                HB = _csc(LH(Basis, Q0B, Q1B, Q2B, jjj0, jjj1, jjj2))
                HG = _csc(LH(Basis, Q0G, Q1G, Q2G, jjj0, jjj1, jjj2))
                HB = HB[keep, :][:, keep]
                HG = HG[keep, :][:, keep]
            else:
                Q0 = _csc(Q0B) + _csc(Q0G)
                if _any(F_['F1']):
                    Q1 = [_csc(Q1B[i]) + _csc(Q1G[i]) for i in range(9)]
                else:
                    Q1 = []
                Q2 = [_csc(Q2B[i]) + _csc(Q2G[i]) for i in range(25)]
                H = _csc(LH(Basis, Q0, Q1, Q2, jjj0, jjj1, jjj2))
                H = H[keep, :][:, keep]
                if saveDiagnostics:
                    diagnostics.Q0 = Q0
                    diagnostics.Q1 = Q1
                    diagnostics.Q2 = Q2
        else:
            Sys.d2psi = F.wignerd(2, float(phiL[iOri]), float(theta[iOri]), 0)
            if explicitFieldSweep:
                EZ0_ = Sys.EZ0
                EZ2_ = Sys.EZ2
                if isfield(Sys, 'NZ0'):
                    NZ0_ = _a(Sys.NZ0).copy()

        for iB in range(B0.size):
            F.logmsg(1, 'Field value %d/%d: %g mT', iB + 1, B0.size, B0[iB])
            if generalLiouvillian:
                if explicitFieldSweep:
                    H = (B0[iB] * HB + HG).tocsc()
                L = (2j * np.pi * H + Gamma).tocsc()
                nDim = L.shape[0]
                if nDim != BasisSize:
                    raise ValueError('Matrix size (%d) inconsistent with basis size (%d). Please report.'
                                     % (nDim, BasisSize))
                if L.shape[1] and _nancols_all(L):
                    raise ValueError('Liouvillian matrix contains NaN entries! Please report.')
            else:
                if explicitFieldSweep:
                    Sys.EZ0 = EZ0_ * B0[iB]
                    Sys.EZ2 = EZ2_ * B0[iB]
                    if isfield(Sys, 'NZ0'):
                        Sys.NZ0 = (NZ0_ * B0[iB]).reshape(np.shape(Sys.NZ0), order='F')
                Sys.DirTilt = Basis.DirTilt
                maxL = len(XLMK) - 1
                if maxL >= 0:
                    xlk = np.zeros((maxL + 1, 2 * maxL + 1), dtype=complex)
                    for L_ in range(maxL + 1):
                        X = np.atleast_2d(np.asarray(XLMK[L_]))
                        xlk[L_, :2 * L_ + 1] = X[L_, :]
                    if not np.any(xlk.imag):
                        xlk = xlk.real
                else:
                    xlk = np.zeros((0, 0))
                Dynamics.xlk = xlk
                Dynamics.maxL = maxL
                Dynamics.Diff = Dynamics.R
                r, c, Vals, nDim = F.chili_lm(Sys, Basis.v, Dynamics, Opt.AllocationBlockSize)
                Vals = np.conj(_a(Vals))
                L = sp.csc_matrix((Vals, (_a(r).astype(np.int64) - 1, _a(c).astype(np.int64) - 1)),
                                  shape=(BasisSize, BasisSize))
                L.sum_duplicates()
                L.eliminate_zeros()
                nDim = int(_f(nDim))
                if saveDiagnostics and iOri == 0:
                    reL = L.real
                    imL = L.imag
                    Gamma = (reL + reL.T) / 2 + 1j * (imL - imL.T) / 2
                    H = (imL + imL.T) / 2 + 1j * (reL.T - reL) / 2
                    H = H / (2 * np.pi)
            if saveDiagnostics and iOri == 0:
                diagnostics.iOri = iOri + 1
                diagnostics.L = L
                diagnostics.H = H
                diagnostics.Gamma = Gamma

            if generalLiouvillian and _if(Opt.useLMKbasis) and useLanczosSolver:
                isComplexSymmetric = _isreal(H)
                if not isComplexSymmetric:
                    L = (TT.conj().T @ L @ TT).tocsc()
                    StartVector = (TT.conj().T @ StartVector).tocsc()

            if _if(Opt.Rescale):
                scale = float(np.max(np.abs(L.data))) if L.nnz else 0.0
                L = L.copy()
                L.data = L.data / scale
                omega = omega0 / scale
            else:
                scale = 1
                omega = omega0
            if config.LogLevel is not None and config.LogLevel >= 2:
                maxDval = float(np.max(np.abs(L.data.real), initial=0))
                F.logmsg(2, '  size: %dx%d, maxabsreal: %g', L.shape[0], L.shape[0], maxDval)
                F.logmsg(2, '  non-zero elements: %d/%d (%0.2f%%)', L.nnz, L.shape[0] ** 2,
                         100 * L.nnz / L.shape[0] ** 2)

            rho0 = _a(StartVector)
            nO = omega.size
            if Opt.Solver == 'L':
                if generalLiouvillian and usePotential:
                    d = L - L.T
                    mx = float(np.max(np.abs(L.data), initial=0))
                    isComplexSymmetric = (float(np.max(np.abs(d.data), initial=0)) / mx) < 1e-10
                    if not isComplexSymmetric:
                        raise ValueError('L is not complex symmetric - cannot use Lanczos method.')
                thisspec, converged, dspec = F.chili_lanczos(L, StartVector, (-1j * omega).reshape(1, -1), Opt)
                thisspec = _a(thisspec)
                if converged:
                    F.logmsg(2, '  converged to within %g at iteration %d/%d', Opt.Threshold, np.size(dspec),
                             BasisSize)
                else:
                    thisspec = np.ones(nO)
                    F.logmsg(0, '  Tridiagonalization did not converge to within %g after %d steps!\n  Increase '
                             'Options.LLMK (current settings [%d,%d,%d,%d])', Opt.Threshold, BasisSize,
                             *[int(x) for x in LLMK])
                thisspec = thisspec / (1 if EASYSPIN_RESCALE else scale)
            elif Opt.Solver == '\\':
                I_ = sp.identity(L.shape[0], format='csc')
                if thisspec is None or np.size(thisspec) < (B0.size if explicitFieldSweep else nO):
                    thisspec = np.zeros(B0.size if explicitFieldSweep else nO, dtype=complex)
                if explicitFieldSweep:
                    Q = (L - 1j * omega[0] * I_).tocsc()
                    thisspec[iB] = np.vdot(rho0, _solve(Q, rho0)) / (1 if EASYSPIN_RESCALE else scale)
                else:
                    for iOmega in range(nO):
                        Q = (L - 1j * omega[iOmega] * I_).tocsc()
                        thisspec[iOmega] = np.vdot(rho0, _solve(Q, rho0)) / (1 if EASYSPIN_RESCALE else scale)
            elif Opt.Solver == 'E':
                Lf = L.toarray()
                Lam, U = np.linalg.eig(Lf)
                Amplitude = (np.conj(rho0) @ U) * np.linalg.solve(U, rho0)
                thisspec = 0
                for iPeak in range(Amplitude.size):
                    thisspec = thisspec + Amplitude[iPeak] / (Lam[iPeak] - 1j * omega)
                thisspec = thisspec / (1 if EASYSPIN_RESCALE else scale)
            elif Opt.Solver == 'B':
                I_ = sp.identity(L.shape[0], format='csc')
                if thisspec is None or np.size(thisspec) < nO:
                    thisspec = np.zeros(nO, dtype=complex)
                for iOmega in range(nO):
                    Q = (L - 1j * omega[iOmega] * I_).tocsc()
                    u, _ = _bicgstab(Q, rho0.astype(complex), _f(Opt.Threshold), nDim)
                    thisspec[iOmega] = np.vdot(rho0, u) / (1 if EASYSPIN_RESCALE else scale)
            elif Opt.Solver == 'C':
                CGshift = 1e-6 + 1e-6j
                _, alpha, beta, err_, stepsDone = F.chili_conjgrad(L, StartVector, CGshift)
                F.logmsg(1, '  step %d/%d: CG converged to within %g', stepsDone, BasisSize, err_)
                thisspec = _a(F.chili_contfracspec((-1j * omega).reshape(1, -1), alpha, beta))
                thisspec = thisspec / (1 if EASYSPIN_RESCALE else scale)
        spec = spec + thisspec * Weights[iOri]

    if EASYSPIN_RESCALE:
        spec = spec / scale
    spec = spec * 1e10
    spec = spec / 2
    if _if(Opt.highField):
        spec = spec / 2
    if Exp.FrequencySweep:
        spec = spec * 1e3
    if saveDiagnostics:
        DIAGNOSTICS[Opt.Diagnostics] = diagnostics
    mwPhase = _f(Exp.mwPhase)
    if not FieldSweep:
        mwPhase = -mwPhase
    spec = np.real(np.exp(1j * mwPhase) * spec).reshape(1, -1)

    if doPostConvolution:
        F.logmsg(1, 'Postconvolution...')
        pcidx = _a(Opt.PostConvNucs).astype(int) - 1
        pcSys = Struct()
        pcSys.g = float(np.mean(np.asarray(fullSys.g, dtype=float).ravel(order='F')))
        Afull = np.atleast_2d(np.asarray(fullSys.A, dtype=float))
        pcSys.A = np.mean(Afull[pcidx, :], axis=1).reshape(-1, 1)
        if isfield(fullSys, 'n'):
            pcSys.n = _a(fullSys.n)[pcidx].reshape(1, -1)
        nucs = fullSys.Nucs if isinstance(fullSys.Nucs, list) else F.nucstring2list(fullSys.Nucs)
        pcSys.Nucs = F.nuclist2string([nucs[i] for i in pcidx])
        pcExp = Struct()
        if FieldSweep:
            pcExp.Range = Exp.Range.reshape(1, -1)
            pcExp.mwFreq = _f(F.unitconvert(np.mean(Exp.Range), 'mT->MHz', pcSys.g)) / 1e3
            Range = Exp.Range
        else:
            pcExp.mwRange = Exp.mwRange.reshape(1, -1)
            pcExp.Field = Exp.Field
            Range = Exp.mwRange * 1e3
        pcExp.Harmonic = 0
        pcExp.nPoints = nPoints
        dx = (Range[1] - Range[0]) / (nPoints - 1)
        pcSys.lw = dx / 12
        spec_pc = _a(F.garlic(pcSys, pcExp, nargout=1))
        spec_pc = spec_pc / np.sum(spec_pc)
        config.LogLevel = Opt.Verbosity
        full = np.convolve(spec.ravel(), spec_pc)
        i0 = spec_pc.size // 2
        spec = full[i0:i0 + spec.size].reshape(1, -1)

    if not isfield(Opt, 'BasisAnalysis'):
        Opt.BasisAnalysis = False
    if _if(Opt.BasisAnalysis):
        F.logmsg(1, '-------------------------------------------------------------------')
        F.logmsg(1, 'Basis set analysis')
        # EasySpin 6.0.12 indexes an undefined variable 'Indices' here (after a bicgstab sweep
        # whose result it never reaches), so the option always fails.
        raise ValueError("Unrecognized function or variable 'Indices'.")

    if np.isfinite(_f(Exp.Temperature)):
        if FieldSweep:
            DeltaE = _c('planck') * Exp.mwFreq * 1e9
        else:
            DeltaE = _c('bmagn') * gavg * Exp.Field * 1e-3
        e = np.exp(-DeltaE / _c('boltzm') / _f(Exp.Temperature))
        Population = np.array([1, e])
        Population = Population / np.sum(Population)
        Polarization = Population[0] - Population[1]
        spec = spec * Polarization

    fwhmG = float(_a(Sys.lw)[0])
    if fwhmG > 0 and ConvolutionBroadening:
        if FieldSweep:
            unitstr = 'mT'
        else:
            unitstr = 'MHz'
            fwhmG = fwhmG / 1e3
        dx = xAxis[1] - xAxis[0]
        F.logmsg(1, 'Convoluting with Gaussian (FWHM %g %s)...', fwhmG, unitstr)
        spec = np.asarray(F.convspec(spec, dx, fwhmG, Exp.ConvHarmonic, 1)).reshape(1, -1)
        Exp.ConvHarmonic = 0

    outspec = spec
    F.logmsg(1, 'Modulation')
    if FieldSweep:
        if ModAmp > 0:
            F.logmsg(1, '  pseudo field modulation with %g mT, harmonic %d', ModAmp, Exp.ModHarmonic)
            outspec = np.asarray(F.fieldmod(xAxis.reshape(1, -1), outspec, ModAmp, Exp.ModHarmonic)).reshape(1, -1)
        else:
            F.logmsg(1, '  no pseudo field modulation')
    else:
        F.logmsg(1, '  no modulation')
    if Exp.DerivHarmonic > 0:
        F.logmsg(1, '  harmonic %d: using differentiation', Exp.DerivHarmonic)
        dx = xAxis[1] - xAxis[0]
        for h in range(int(Exp.DerivHarmonic)):
            dspec = np.diff(outspec, axis=1) / dx
            outspec = (dspec[:, np.r_[0, 0:dspec.shape[1]]] + dspec[:, np.r_[0:dspec.shape[1], dspec.shape[1] - 1]]) / 2
    else:
        F.logmsg(1, '  no derivative')

    xAxis = np.asarray(xAxis).reshape(1, -1)
    if plot:
        F.cwepr_plot(xAxis, outspec, Exp)
    F.logmsg(1, '-------------------------------------------------------------------')
    if nargout == 0:
        return None
    if nargout == 1:
        return outspec
    if nargout == 2:
        return xAxis, outspec
    return xAxis, outspec, Struct()


def _nancols_all(L):
    """MATLAB `if any(isnan(L))` on a matrix: true only if every column holds a NaN."""
    Lc = L.tocsc()
    nan = np.isnan(Lc.data)
    if not nan.any():
        return False
    cols = np.repeat(np.arange(Lc.shape[1]), np.diff(Lc.indptr))
    return np.unique(cols[nan]).size == Lc.shape[1]
