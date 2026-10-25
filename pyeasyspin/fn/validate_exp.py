"""Validate the experiment structure Exp (field/frequency sweep, range, harmonic, ...)"""
import numpy as np
from .._registry import F, nout
from .._struct import Struct


def _first(x):
    return x[0] if isinstance(x, tuple) else x


def _arr(x):
    return np.atleast_1d(np.asarray(x, dtype=float)).ravel(order='F')


def _isreal(x):
    return not np.iscomplexobj(np.asarray(x))


def _empty(x):
    return x is None or (hasattr(x, '__len__') and len(x) == 0) or (isinstance(x, np.ndarray) and x.size == 0)


def _autorange_hf(Sys, Exp, cardamom_like):
    I = np.ravel(np.asarray(F.nucspin(Sys.Nucs), dtype=float))
    if I.size > 0:
        Amax = np.max(np.abs(np.atleast_2d(np.asarray(Sys.A, dtype=float))), axis=1)
        hf = np.sum(I * Amax) * 1e6
    else:
        hf = 0
    return hf


def _sweep_center(Sys, Exp, FieldSweep):
    """Shared chili/cardamom part: CenterSweep/Range and mwCenterSweep/mwRange."""
    if FieldSweep:
        if 'CenterSweep' in Exp:
            if 'Range' in Exp:
                F.logmsg(0, 'Using Exp.CenterSweep and ignoring Exp.Range.')
        else:
            if 'Range' in Exp:
                R = _arr(Exp.Range)
                Exp.CenterSweep = np.array([[np.mean(R), np.diff(R)[0]]])
            else:
                if Sys.nElectrons == 1 and Sys.S == 0.5:
                    F.logmsg(1, '  automatic determination of sweep range')
                    Stretch = 1.25
                    hf = _autorange_hf(Sys, Exp, True)
                    g = np.asarray(Sys.g, dtype=float)
                    gmax, gmin = np.max(g), np.min(g)
                    planck, bmagn = _first(F.planck()), _first(F.bmagn())
                    minB = planck * (Exp.mwFreq * 1e9 - hf) / bmagn / gmax / 1e-3
                    maxB = planck * (Exp.mwFreq * 1e9 + hf) / bmagn / gmin / 1e-3
                    Exp.CenterSweep = np.array([[(maxB + minB) / 2, Stretch * max(maxB - minB, 5)]])
                else:
                    raise ValueError('Cannot automatically determine sweep range for this spin system.')
    else:
        if 'mwCenterSweep' in Exp:
            if 'mwRange' in Exp:
                F.logmsg(0, 'Using Exp.mwCenterSweep and ignoring Exp.mwRange.')
        else:
            if 'mwRange' in Exp:
                R = _arr(Exp.mwRange)
                Exp.mwCenterSweep = np.array([[np.mean(R), np.diff(R)[0]]])
            else:
                raise ValueError('Either Exp.mwRange or Exp.mwCenterSweep need to be given.')
    return Exp


def _fieldfreq(Exp, strictField):
    if 'mwFreq' not in Exp:
        if 'Field' not in Exp:
            raise ValueError('Please supply either the microwave frequency in Exp.mwFreq (for field sweeps) or the '
                             'magnetic field in Exp.Field (for frequency sweeps).')
        FieldSweep = False
    else:
        if 'Field' in Exp:
            raise ValueError('Give either Exp.mwFreq (for a field sweep) or Exp.Field (for a frequency sweep), but not both.')
        FieldSweep = True
    if FieldSweep:
        f = np.asarray(Exp.mwFreq)
        if f.size != 1 or np.any(f <= 0) or not _isreal(f):
            raise ValueError('Uninterpretable microwave frequency in Exp.mwFreq.')
        F.logmsg(1, '  field sweep, mw frequency %0.8g GHz', Exp.mwFreq)
    else:
        b = np.asarray(Exp.Field)
        if b.size != 1 or (np.any(b <= 0) if strictField else np.any(b < 0)) or not _isreal(b):
            raise ValueError('Uninterpretable magnetic field in Exp.Field.')
        F.logmsg(1, '  frequency sweep, magnetic field %0.8g mT', Exp.Field)
    return FieldSweep


def _harmonic_modamp(Exp, FieldSweep, msg_harm):
    h = Exp.Harmonic
    if _empty(h) or 'Harmonic' not in Exp or np.any(np.isnan(np.asarray(h, dtype=float))):
        Exp.Harmonic = 1 if FieldSweep else 0
    if not np.any(np.asarray(Exp.Harmonic) == np.array([-1, 0, 1, 2])):
        raise ValueError(msg_harm)
    ma = np.asarray(Exp.ModAmp, dtype=float)
    if np.any(ma < 0) or np.any(np.isnan(ma)) or ma.size != 1:
        raise ValueError('Exp.ModAmp must be either a single positive number or zero.')
    if Exp.ModAmp > 0:
        if FieldSweep:
            F.logmsg(1, '  field modulation, amplitude %g mT', Exp.ModAmp)
            if Exp.Harmonic < 1:
                raise ValueError('With field modulation (Exp.ModAmp), Exp.Harmonic=0 does not work.')
            Exp.ModHarmonic = Exp.Harmonic
            Exp.ConvHarmonic = 0
            Exp.DerivHarmonic = 0
        else:
            raise ValueError('Exp.ModAmp cannot be used with frequency sweeps.')
    else:
        Exp.ModHarmonic = 0
        # EasySpin reads the undefined variable ConvolutionBroadening here
        raise NameError("Unrecognized function or variable 'ConvolutionBroadening'.")
    return Exp


@nout(-1)
def validate_exp(program=None, Sys=None, Exp=None):
    """validate_exp(program, Sys, Exp). 'cardamom' -> (Exp, FieldSweep, CenterField, CenterFreq, Sweep)
    (CenterFreq = [] for field sweeps). The 'pepper' and 'chili' branches read variables EasySpin 6.0.12
    never defines (StrainWidths, ConvolutionBroadening, usePotential) and fail as in MATLAB; pepper
    and chili have their own validation."""
    if not isinstance(program, str):
        raise ValueError('Program name must be a string.')
    Exp = Struct(Exp)
    if program == 'pepper':
        Default = Struct(CenterSweep=np.nan, Range=np.nan, mwCenterSweep=np.nan, mwRange=np.nan, nPoints=1024,
                         Temperature=np.nan, Harmonic=np.nan, mwMode='', Ordering=np.zeros((0, 0)), ModAmp=0,
                         mwPhase=0, SampleFrame=np.zeros((0, 0)), CrystalSymmetry='', MolFrame=np.zeros((0, 0)))
        Exp = F.adddefaults(Exp, Default)
        if 'mwFreq' not in Exp or _empty(Exp.mwFreq):
            if 'Field' not in Exp:
                raise ValueError('Please supply either the microwave frequency in Exp.mwFreq (for field sweeps) or the '
                                 'magnetic field in Exp.Field (for frequency sweeps).')
            FieldSweep = False
        else:
            if 'Field' in Exp and not _empty(Exp.Field):
                raise ValueError('Give either Exp.mwFreq (for a field sweep) or Exp.Field (for a frequency sweep), but not both.')
            FieldSweep = True
        if FieldSweep:
            f = np.asarray(Exp.mwFreq)
            if f.size != 1 or np.any(f <= 0) or not _isreal(f):
                raise ValueError('Uninterpretable microwave frequency in Exp.mwFreq.')
            F.logmsg(1, '  field sweep, mw frequency %0.8g GHz', Exp.mwFreq)
        else:
            b = np.asarray(Exp.Field)
            if b.size != 1 or np.any(b < 0) or not _isreal(b):
                raise ValueError('Uninterpretable magnetic field in Exp.Field.')
            F.logmsg(1, '  frequency sweep, magnetic field %0.8g mT', Exp.Field)
        if FieldSweep and np.all(np.isnan(_arr(Exp.CenterSweep))) and np.all(np.isnan(_arr(Exp.Range))):
            Lx = Sys.L if 'L' in Sys else np.zeros((0, 0))
            if np.size(Sys.S) == 1 and Sys.S == 0.5 and not np.any(np.asarray(Lx)):
                F.logmsg(1, '  automatic determination of sweep range')
                I = np.ravel(np.asarray(F.nucspin(Sys.Nucs), dtype=float))
                if I.size:
                    Amax = np.max(np.abs(np.asarray(Sys.A, dtype=float)), axis=1)
                    if Sys.fullA:
                        Amax = np.max(Amax.reshape(3, -1, order='F'), axis=0)
                else:
                    Amax = 0
                hf = np.sum(I * Amax) * 1e6
                if Sys.fullg:
                    gg = np.asarray(Sys.g, dtype=float)
                    g = np.concatenate([np.linalg.eigvalsh(gg[3 * k:3 * k + 3, :]) for k in range(int(Sys.nElectrons))])
                else:
                    g = np.ravel(np.asarray(Sys.g, dtype=float), order='F')
                planck, bmagn = _first(F.planck()), _first(F.bmagn())
                minB = planck * (Exp.mwFreq * 1e9 - hf) / bmagn / np.max(g) / 1e-3
                maxB = planck * (Exp.mwFreq * 1e9 + hf) / bmagn / np.min(g) / 1e-3
                Center = (maxB + minB) / 2
                Sweep = maxB - minB
                if Sweep == 0:
                    Sweep = 5 * np.max(Sys.lw)
                if Sweep == 0:
                    Sweep = 10
                Exp.CenterSweep = np.array([[Center, 1.25 * Sweep]])
            else:
                raise ValueError('Cannot automatically determine field range.\nPlease given either Exp.CenterSweep or Exp.Range.')
        if FieldSweep:
            if not np.any(np.isnan(_arr(Exp.CenterSweep))):
                cs = _arr(Exp.CenterSweep)
                Exp.Range = np.maximum(cs[0] + np.array([[-1, 1]]) * cs[1] / 2, 0)
            R = _arr(Exp.Range)
            if np.all(~np.isnan(R)):
                if np.any(np.diff(R) <= 0) or np.any(~np.isfinite(R)) or np.any(R < 0):
                    raise ValueError('Exp.Range is not valid!')
        else:
            if not np.any(np.isnan(_arr(Exp.mwCenterSweep))):
                cs = _arr(Exp.mwCenterSweep)
                Exp.mwRange = np.maximum(cs[0] + np.array([[-1, 1]]) * cs[1] / 2, 0)
            R = _arr(Exp.mwRange)
            if np.all(~np.isnan(R)):
                if np.any(np.diff(R) <= 0) or np.any(~np.isfinite(R)) or np.any(R < 0):
                    raise ValueError('Exp.mwRange is not valid!')
        n = np.asarray(Exp.nPoints)
        if np.iscomplexobj(n) or n.size > 1 or Exp.nPoints < 2:
            raise ValueError('Problem with Exp.nPoints. Needs to be a number not smaller than 2.')
        if FieldSweep:
            F.logmsg(1, '  frequency %g GHz, field range [%g %g] mT, %d points', Exp.mwFreq, _arr(Exp.Range)[0],
                     _arr(Exp.Range)[1], Exp.nPoints)
            raise NameError("Unrecognized function or variable 'StrainWidths'.")
        raise NameError("Unrecognized function or variable 'SweepAutoRange'.")
    elif program == 'chili':
        if 'MOMD' in Exp:
            raise ValueError('Exp.MOMD is obsolete. Remove it from your code. See the documentation for details.')
        for k, v in (('nPoints', 1024), ('Harmonic', np.zeros((0, 0))), ('mwPhase', 0), ('Temperature', np.nan),
                     ('ModAmp', 0), ('mwMode', 'perpendicular'), ('Ordering', np.zeros((0, 0))),
                     ('SampleFrame', np.zeros((0, 0)))):
            if k not in Exp:
                Exp[k] = v
        n = np.asarray(Exp.nPoints)
        if np.iscomplexobj(n) or n.size > 1 or Exp.nPoints < 2:
            raise ValueError('Problem with Exp.nPoints. Needs to be a number >= 2.')
        T = np.asarray(Exp.Temperature, dtype=float)
        if not np.all(np.isnan(T)):
            if T.size != 1 or np.isinf(T).any() or (T < 0).any():
                raise ValueError('Problem with Exp.Temperature. If given, Exp.Temperature must be a positive value.')
        FieldSweep = _fieldfreq(Exp, True)
        Exp = _sweep_center(Sys, Exp, FieldSweep)
        if FieldSweep:
            cs = _arr(Exp.CenterSweep)
            Exp.Range = cs[0] + np.array([[-1, 1]]) / 2 * cs[1]
            if np.any(Exp.Range < 0) or np.diff(_arr(Exp.Range))[0] <= 0:
                raise ValueError('Invalid sweep range! Check Exp.CenterSweep or Exp.Range.')
        else:
            cs = _arr(Exp.mwCenterSweep)
            Exp.mwRange = cs[0] + np.array([[-1, 1]]) / 2 * cs[1]
            if np.any(Exp.mwRange < 0) or np.diff(_arr(Exp.mwRange))[0] <= 0:
                raise ValueError('Invalid sweep range! Check Exp.mwCenterSweep or Exp.mwRange.')
        _harmonic_modamp(Exp, FieldSweep, 'Exp.Harmonic must be 0, 1 or 2.')
        if Exp.mwMode not in ('perpendicular', 'parallel'):
            raise ValueError("Exp.mwMode must be either 'perpendicular' or 'parallel'.")
        F.logmsg(1, '  harmonic %d, %s mode', Exp.Harmonic, Exp.mwMode)
        if 'CrystalSymmetry' in Exp:
            import warnings
            warnings.warn('Exp.CrystalSymmetry is not used by chili.')
        if not _empty(Exp.Ordering):
            o = Exp.Ordering
            if not callable(o) and not (np.size(o) == 1 and _isreal(o)):
                raise ValueError('Exp.Ordering must be a single number or a function handle.')
        raise NameError("Unrecognized function or variable 'usePotential'.")
    elif program == 'cardamom':
        for k, v in (('nPoints', 1024), ('Harmonic', np.zeros((0, 0))), ('mwPhase', 0), ('Temperature', np.nan),
                     ('ModAmp', 0), ('mwMode', 'perpendicular'), ('Ordering', np.zeros((0, 0))),
                     ('SampleFrame', np.zeros((0, 0)))):
            if k not in Exp:
                Exp[k] = v
        n = np.asarray(Exp.nPoints)
        if np.iscomplexobj(n) or n.size > 1 or Exp.nPoints < 2:
            raise ValueError('Problem with Exp.nPoints. Needs to be a number >= 2.')
        FieldSweep = _fieldfreq(Exp, True)
        Exp = _sweep_center(Sys, Exp, FieldSweep)
        if FieldSweep:
            CenterFreq = np.zeros((0, 0))
            cs = _arr(Exp.CenterSweep)
            CenterField = float(cs[0])
            Sweep = float(cs[1])
            Exp.Range = cs[0] + np.array([[-1, 1]]) / 2 * Sweep
            if np.any(Exp.Range < 0) or np.diff(_arr(Exp.Range))[0] <= 0:
                raise ValueError('Invalid sweep range! Check Exp.CenterSweep or Exp.Range.')
        else:
            cs = _arr(Exp.mwCenterSweep)
            CenterFreq = float(cs[0])
            Sweep = float(cs[1])
            Exp.mwRange = cs[0] + np.array([[-1, 1]]) / 2 * Sweep
            CenterField = Exp.Field
            if np.any(Exp.mwRange < 0) or np.diff(_arr(Exp.mwRange))[0] <= 0:
                raise ValueError('Invalid sweep range! Check Exp.mwCenterSweep or Exp.mwRange.')
        if FieldSweep:
            R = _arr(Exp.Range)
            F.logmsg(1, '  field range (mT): min %g, max %g, center %g, width %g', R[0], R[1], CenterField, Sweep)
        else:
            R = _arr(Exp.mwRange)
            F.logmsg(1, '  frequency range (GHz): min %g, max %g, center %g, width %g', R[0], R[1], CenterFreq, Sweep)
        return Exp, FieldSweep, CenterField, CenterFreq, Sweep
    else:
        raise ValueError('Program not recognized.')
