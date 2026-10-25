"""Spin system, initial state, relaxation, excitation and detection operators for spidyan/saffron."""
import warnings

import numpy as np

from .._registry import F, nout
from .._struct import Struct
from .._mat import isempty, vec


def _any(x):
    return bool(np.any(vec(np.asarray(x, dtype=float)) != 0)) if not isempty(x) else False


def _dense(x):
    return x.toarray() if hasattr(x, 'toarray') else np.asarray(x)


@nout(5)
def s_propagationsetup(Sys=None, Events=None, Opt=None):
    """(Sys, Sigma0, DetOps, Events, Relaxation): validated Sys (with T1/T2 defaults), initial
    density matrix (-Sz total, or Sys.initState{1}), list of detection operators (default S+ of all
    electrons), Events with excitation operators xOp (Sx total, +Sy for circular excitation, or
    Opt.ExcOperator), and Relaxation = Struct(Gamma, equilibriumState) or [] without T1/T2."""
    Sys = Struct(Sys)
    Opt = Struct(Opt) if Opt is not None else Struct()
    Events = [Struct(e) for e in Events] if isinstance(Events, list) else [Struct(Events)]
    if 'Nucs' not in Sys:
        Sys.Nucs = ''
    Sys, err = F.validatespinsys(Sys)
    if not isempty(err):
        raise ValueError(err)
    Sys = Struct(Sys)
    if bool(Sys.MO_present):
        raise ValueError('saffron/spidyan do not support general parameters!')
    if _any(Sys.L):
        raise ValueError('saffron/spidyan do not support L!')
    if Sys.nNuclei == 0:
        F.logmsg(1, '  spin system: %d electron(s)', Sys.nElectrons)
    else:
        F.logmsg(1, '  spin system: %d electron(s), %d nuclei', Sys.nElectrons, Sys.nNuclei)
    if 'n' in Sys and np.any(vec(np.asarray(Sys.n, dtype=float)) != 1):
        raise ValueError('saffron/spidyan do not support Sys.n. Specify equivalent nuclei explicitly.')
    if 'nn' in Sys and np.any(vec(np.asarray(Sys.nn, dtype=float)) != 0):
        raise ValueError('saffron/spidyan do not support nuclear-nuclear couplings (Sys.nn).')
    F.logmsg(1, '  looking for and verifying fields related to relaxation')
    if 'eqState' in Sys and 'T2' not in Sys and 'T1' not in Sys:
        warnings.warn('An equilibrium state was provided, but no relaxation times. Equilibrium state ignored.')
    if 'T1' not in Sys and 'T2' not in Sys:
        if 'T1T2' in Sys:
            raise ValueError('T1 and T2 should be provided in Sys.T1 and Sys.T2 and not in Sys.T1T2.')
        Sys.T1 = 0
        Sys.T2 = 0
    elif 'T1' in Sys and 'T2' not in Sys:
        Sys.T2 = 0
    elif 'T2' in Sys and 'T1' not in Sys:
        Sys.T1 = 0
    T12 = np.concatenate([vec(np.asarray(Sys.T1)), vec(np.asarray(Sys.T2))])
    if np.any(np.real(T12) < 0) or np.iscomplexobj(T12):
        raise ValueError('T1 and T2 in Sys.T1 and Sys.T2 must be positive and in microseconds.')

    F.logmsg(1, '  obtaining spin operators')
    totSpinOps = [None, None, None]
    for iSpin in range(1, np.size(Sys.S) + 1):
        for i in range(3):
            op = F.sop(Sys.Spins, np.array([[iSpin, i + 1]], dtype=float))
            totSpinOps[i] = op if iSpin == 1 else totSpinOps[i] + op

    F.logmsg(1, '  setting up the initial state')
    if 'initState' in Sys and not isempty(Sys.initState):
        if Sys.initState[1] != 'uncoupled':
            raise ValueError('spidyan only accepts a density matrix in the uncoupled basis in Sys.initState.')
        a, b = np.atleast_2d(_dense(Sys.initState[0])).shape[:2]
        if Sys.nStates != a or Sys.nStates != b:
            raise ValueError('Initial state has to be a density matrix.')
        Sigma0 = Sys.initState[0]
    else:
        Sigma0 = -totSpinOps[2]
    if np.all(T12 == 0):
        Relaxation = np.zeros((0, 0))
    else:
        F.logmsg(1, '  setting up the relaxation superoperator and the equilibrium state')
        Relaxation = Struct()
        Relaxation.Gamma = F.s_relaxationsuperoperator(Sys)
        if 'eqState' in Sys and not isempty(Sys.eqState):
            if isinstance(Sys.eqState, str):
                raise ValueError('String input for equilibrium state not yet supported.')
            a, b = np.shape(np.atleast_2d(Sys.eqState))[:2]
            if Sys.nStates != a or Sys.nStates != b:
                raise ValueError('User-provided equilibrium state does not have the same dimensions as the density matrix.')
            Relaxation.equilibriumState = Sys.eqState
        else:
            Relaxation.equilibriumState = Sigma0
    if 'Relaxation' in Opt and np.any(np.asarray(Opt.Relaxation) != 0) and isempty(Relaxation):
        raise ValueError('You need to provide relaxation times Sys.T1 and Sys.T2 if you request relaxation.')

    F.logmsg(1, '  setting up the excitation operators')
    ExcOp = Opt.get('ExcOperator', None)
    if ExcOp is not None and not isinstance(ExcOp, list):
        ExcOp = [ExcOp]
    iPulse = 1
    for iEvent in range(len(Events)):
        if Events[iEvent].type == 'pulse':
            if (ExcOp is not None and not isempty(ExcOp) and iPulse <= len(ExcOp)
                    and not isempty(ExcOp[iPulse - 1])):
                x = ExcOp[iPulse - 1]
                if isinstance(x, str):
                    Events[iEvent].xOp = F.sop(Sys.Spins, x)
                elif np.shape(_dense(x)) != np.shape(_dense(Sigma0)):
                    raise ValueError('The excitation operator that you provided for pulse no. ' + str(iPulse)
                                     + ' does not have the same size as the density matrix.')
                else:
                    Events[iEvent].xOp = x
            else:
                Events[iEvent].xOp = totSpinOps[0]
                if bool(Events[iEvent].ComplexExcitation):
                    Events[iEvent].xOp = Events[iEvent].xOp + totSpinOps[1]
            iPulse += 1

    F.logmsg(1, '  setting up the detection operators')
    useDefault = 'DetOperator' not in Opt or isempty(Opt.DetOperator)
    if useDefault:
        DetOps = [totSpinOps[0] + 1j * totSpinOps[1]]
    else:
        DetOp = Opt.DetOperator
        if isinstance(DetOp, str):
            DetOp = [DetOp]
        elif not isinstance(DetOp, list):
            DetOp = [DetOp]
        DetOps = [None] * len(DetOp)
        for k, d in enumerate(DetOp):
            if isinstance(d, str):
                DetOps[k] = F.sop(Sys.Spins, d)
            else:
                if np.shape(_dense(d)) != np.shape(_dense(totSpinOps[0])):
                    raise ValueError('The size of the provided detection operator does not match the defined spin system.')
                DetOps[k] = d
    F.logmsg(1, '  spin system validation finished successfully!')
    return Sys, Sigma0, DetOps, Events, Relaxation
