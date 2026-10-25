"""Pulse sequence definitions of the predefined experiments for the thyme method."""
import math

import numpy as np

from .._struct import Struct


def _field(Exp, name):
    if name not in Exp:
        raise ValueError(f'Unrecognized field name "{name}".')
    return Exp[name]


def _checkfields(requiredFields, Exp):
    for f in requiredFields:
        if f not in Exp:
            raise ValueError('The field Exp.' + f + ' is required to run ' + Exp['Sequence']
                             + ' with the general method (and may not be allowed to be zero).')


def _detwindow(Exp):
    if 'DetWindow' not in Exp:
        if 'DetSequence' in Exp:
            # EasySpin: Exp = rmfield(Exp.DetSequence) -- rmfield needs two arguments
            raise ValueError('Not enough input arguments.')
        Exp.DetWindow = 0


def s_predefinedexperiments(Exp=None):
    """Exp with Exp.Sequence ('2pESEEM', '3pESEEM' or 'HYSCORE') replaced by a list of ideal
    (rectangular, tp 16/32 ns) pulses and delays, plus Exp.Dim1/Dim2, Exp.nPoints, Exp.DetWindow
    and, for 3pESEEM, a two-step phase cycle of pulses 1 and 3."""
    Exp = Struct(Exp)
    defaultPoints = 512
    P90 = Struct(tp=0.016, Flip=math.pi / 2)
    P180 = Struct(tp=0.032, Flip=math.pi)
    seq = Exp.Sequence
    if seq == '2pESEEM':
        _checkfields(['dt', 'tau'], Exp)
        _detwindow(Exp)
        Exp.Sequence = [Struct(P90), Exp.tau, Struct(P180), Exp.tau]
        if 'nPoints' not in Exp:
            Exp.nPoints = defaultPoints
        Exp.Dim1 = ['d1,d2', Exp.dt]
    elif seq == '3pESEEM':
        _checkfields(['dt', 'tau', 'T'], Exp)
        _detwindow(Exp)
        Exp.Sequence = [Struct(P90), Exp.tau, Struct(P90), Exp.T, Struct(P90), Exp.tau]
        if 'nPoints' not in Exp:
            Exp.nPoints = defaultPoints
        Exp.Dim1 = ['d2', Exp.dt]
        pc = list(Exp.PhaseCycle) if 'PhaseCycle' in Exp and isinstance(Exp.PhaseCycle, list) else []
        while len(pc) < 3:
            pc.append(np.zeros((0, 0)))
        pc[0] = np.array([[0, 1], [math.pi, -1]], dtype=float)
        pc[2] = np.array([[0, 1], [math.pi, -1]], dtype=float)
        Exp.PhaseCycle = pc
    elif seq == 'HYSCORE':
        _checkfields(['t1', 't2', 'tau'], Exp)
        _detwindow(Exp)
        Exp.Sequence = [Struct(P90), Exp.tau, Struct(P90), Exp.t1, Struct(P180), Exp.t2, Struct(P90), Exp.tau]
        if 'nPoints' not in Exp:
            Exp.nPoints = np.array([[defaultPoints / 4, defaultPoints / 4]])
        elif np.size(Exp.nPoints) == 1:
            n = float(np.ravel(Exp.nPoints)[0])
            Exp.nPoints = np.array([[n, n]])
        Exp.Dim1 = ['d2', _field(Exp, 'dt')]
        Exp.Dim2 = ['d3', _field(Exp, 'dt')]
    return Exp
