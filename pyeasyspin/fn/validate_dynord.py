"""Validate the dynamics and ordering parameters (Sys.tcorr, Sys.Diff, potential, ...) for chili,
cardamom and stochtraj_diffusion"""
import numpy as np
from .._registry import nout
from .._struct import Struct


def _copyfields(dst, src, names):
    for n in names:
        if n in src:
            dst[n] = src[n]


def _orient_like(D, idx):
    """MATLAB D(idx): a scalar gives a row; a vector keeps its orientation."""
    a = np.asarray(D, dtype=float)
    v = a.ravel(order='F')[list(idx)]
    if a.ndim == 2 and a.shape[1] == 1 and a.shape[0] > 1:
        return v.reshape(-1, 1)
    return v.reshape(1, -1)


def _processdynamics(D, FieldSweep, isDiffSim=None):
    if isDiffSim is None:
        raise ValueError('Not enough input arguments.')
    Dyn = Struct(D)
    err = ''
    if isDiffSim:
        if 'Diff' in Dyn:
            pass
        elif 'logDiff' in Dyn:
            Dyn['Diff'] = 10.0 ** np.asarray(Dyn['logDiff'], dtype=float)
        elif 'tcorr' in Dyn:
            Dyn['Diff'] = 1 / 6 / np.asarray(Dyn['tcorr'], dtype=float)
        elif 'logtcorr' in Dyn:
            if np.all(np.asarray(Dyn['logtcorr']) >= 0):
                raise ValueError('Sys.logtcorr must be negative.')
            Dyn['Diff'] = 1 / 6 / 10.0 ** np.asarray(Dyn['logtcorr'], dtype=float)
        else:
            err = ('You must specify a rotational correlation time or a diffusion tensor\n'
                   '(Sys.tcorr, Sys.logtcorr, Sys.Diff or Sys.logDiff).')
            return Dyn, err
        Diff = np.asarray(Dyn['Diff'], dtype=float)
        if np.any(Diff < 0):
            raise ValueError('Negative diffusion rate or correlation times are not possible.')
        elif np.any(Diff > 1e12):
            print('Diffusion rate very fast. Simulation might not converge.')
        elif np.any(Diff < 1e3):
            print('Diffusion rate very slow. Simulation might not converge.')
        n = Diff.size
        if n == 1:
            Dyn['Diff'] = _orient_like(Diff, [0, 0, 0])
        elif n == 2:
            Dyn['Diff'] = _orient_like(Diff, [0, 0, 1])
        elif n == 3:
            pass
        else:
            err = 'Sys.Diff must have 1, 2 or 3 elements (isotropic, axial, rhombic).'
            return Dyn, err
        if np.ndim(Dyn['Diff']) == 0:
            Dyn['Diff'] = float(Dyn['Diff'])
    if 'lw' in Dyn:
        lw = np.asarray(Dyn['lw'], dtype=float).ravel(order='F')
        if lw.size > 1:
            if FieldSweep is not None and np.size(FieldSweep) > 0 and np.all(FieldSweep):
                LorentzFWHM = lw[1] * 28 * 1e6
            else:
                LorentzFWHM = lw[1] * 1e6
        else:
            LorentzFWHM = 0
        if LorentzFWHM != 0:
            Dyn['T2'] = 1 / LorentzFWHM / np.pi
        else:
            Dyn['T2'] = np.inf
    if 'Exchange' not in Dyn:
        Dyn['Exchange'] = 0
    Dyn['Exchange'] = Dyn['Exchange'] * 2 * np.pi * 1e6
    return Dyn, err


@nout(-1)
def validate_dynord(program=None, Sys=None, FieldSweep=None, isDiffSim=None):
    """validate_dynord(program, Sys[, FieldSweep, isDiffSim]) -> Dynamics ('cardamom') or
    (Dynamics, Sim) ('stochtraj_diffusion'). The 'chili' branch fails as in EasySpin 6.0.12."""
    if not isinstance(program, str):
        raise ValueError('Program name must be a string.')
    if FieldSweep is None:
        FieldSweep = False
        isDiffSim = True
    elif isDiffSim is None:
        isDiffSim = True
    Sys = Struct(Sys)
    if program == 'chili':
        if 'psi' in Sys:
            raise ValueError('Sys.psi is obsolete. Remove it from your code. See the documentation for details.')
        Sys.setdefault('DiffFrame', np.array([[0.0, 0, 0]]))
        Sys.setdefault('Exchange', 0)
        Sys.setdefault('Potential', np.zeros((0, 0)))
        Dynamics = Struct()
        _copyfields(Dynamics, Sys, ['tcorr', 'Diff', 'logtcorr', 'logDiff', 'lwpp', 'lw'])
        Dynamics['Exchange'] = Sys['Exchange']
        Dynamics, err = _processdynamics(Dynamics, FieldSweep)   # EasySpin passes no isDiffSim here
        if err:
            raise ValueError(err)
        raise ValueError("Unrecognized function or variable 'Potential'.")
    elif program == 'cardamom':
        if 'DiffGlobal' not in Sys:
            Sys['DiffGlobal'] = np.zeros((0, 0))
        Dynamics = Struct()
        if isDiffSim:
            _copyfields(Dynamics, Sys, ['tcorr', 'Diff', 'logtcorr', 'logDiff'])
        _copyfields(Dynamics, Sys, ['DiffGlobal', 'lwpp', 'lw'])
        Dynamics, err = _processdynamics(Dynamics, FieldSweep, isDiffSim)
        if err:
            raise ValueError(err)
        return Dynamics
    elif program == 'stochtraj_diffusion':
        Sim = Struct()
        if 'Potential' in Sys:
            P = Sys['Potential']
            hasl = isinstance(P, dict) and 'lambda' in P
            hasLMK = isinstance(P, dict) and 'LMK' in P
            if hasl and hasLMK:
                LMK = np.atleast_2d(np.asarray(P['LMK'], dtype=float))
                lam = np.atleast_2d(np.asarray(P['lambda']))
                if LMK.ndim != 2 or LMK.shape[1] != 3:
                    raise ValueError('LMK must be an array of shape Nx3.')
                if lam.ndim != 2 or lam.shape[1] != 2:
                    raise ValueError('lambda must be an array of shape Nx2.')
                for j in range(LMK.shape[0]):
                    L, M, K = LMK[j]
                    if not L > 0:
                        raise ValueError('For all sets of indices LMK, it is required that L>0.')
                    if K == 0:
                        if not (0 <= M and M <= L):
                            raise ValueError('For all sets of indices LMK, if K=0, then it is required that 0<=M<=L.')
                    else:
                        if not (0 < K and K <= L and abs(M) <= L):
                            raise ValueError('For all sets of indices LMK, if K~=0, then it is required that 0<K<=L and |M|<=L.')
                Sim['lambda'] = P['lambda']
                Sim['LMK'] = P['LMK']
            elif not hasl and not hasLMK:
                Sim['lambda'] = np.zeros((0, 0))
                Sim['LMK'] = np.zeros((0, 0))
            else:
                raise ValueError('Both potential coefficients and LMK are required for an orientational potential.')
        else:
            Sim['lambda'] = np.zeros((0, 0))
            Sim['LMK'] = np.zeros((0, 0))
        Dynamics = Struct()
        _copyfields(Dynamics, Sys, ['tcorr', 'Diff', 'logtcorr', 'logDiff'])
        if not Dynamics:
            raise ValueError("Unrecognized function or variable 'Dynamics'.")
        Dynamics, err = _processdynamics(Dynamics, np.zeros((0, 0)), isDiffSim)
        if err:
            raise ValueError(err)
        return Dynamics, Sim
    else:
        raise ValueError('Program not recognized.')
