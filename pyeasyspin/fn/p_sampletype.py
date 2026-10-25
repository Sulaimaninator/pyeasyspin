"""Sample type (crystal, disordered, partially ordered) and sample rotations."""
import inspect

import numpy as np

from .._registry import F, nout
from .._struct import Struct
from .._mat import isempty, isnumeric, numel


def _nargin(f):
    try:
        return len(inspect.signature(f).parameters)
    except (TypeError, ValueError):
        return -1


@nout(2)
def p_sampletype(Exp=None, Opt=None):
    """(Exp, Opt) with CrystalSymmetry/MolFrame/SampleFrame defaults, Exp.Ordering as a function
    of (alpha, beta, gamma), and Opt.R_L2S, rotatedSample and the three sample-type flags."""
    Exp, Opt = Struct(Exp), Struct(Opt)
    partiallyOrderedSample = not isempty(Exp.Ordering)
    crystalSample = not partiallyOrderedSample and (not isempty(Exp.MolFrame) or not isempty(Exp.CrystalSymmetry))
    disorderedSample = not partiallyOrderedSample and not crystalSample
    if partiallyOrderedSample:
        if not isempty(Exp.MolFrame):
            raise ValueError('Exp.MolFrame cannot be used for partially ordered samples (Exp.Ordering given).')
        if not isempty(Exp.CrystalSymmetry):
            raise ValueError('Exp.CrystalSymmetry cannot be used for partially ordered samples (Exp.Ordering given).')
    if crystalSample:
        if isempty(Exp.CrystalSymmetry):
            Exp.CrystalSymmetry = 'P1'
        if isempty(Exp.MolFrame):
            Exp.MolFrame = np.zeros((1, 3))
    if isempty(Exp.SampleFrame):
        Exp.SampleFrame = np.zeros((1, 3))
    if not isnumeric(Exp.SampleFrame):
        raise ValueError('Exp.SampleFrame must be an Nx3 array.')
    Opt.rotatedSample = (bool(np.any(np.asarray(Exp.SampleFrame) != 0))
                         or ('SampleRotation' in Exp and not isempty(Exp.SampleRotation)))
    if partiallyOrderedSample:
        o = Exp.Ordering
        if isnumeric(o) and numel(o) == 1 and not np.iscomplexobj(o):
            lam = float(np.asarray(o, dtype=float).ravel()[0])
            Exp.Ordering = lambda alpha, beta, gamma: np.exp(lam * F.plegendre(2, 0, np.cos(beta)))
            F.logmsg(1, '  partial ordering (built-in function, coefficient = %g)', lam)
        elif callable(o):
            F.logmsg(1, '  partial ordering (user-supplied function)')
        else:
            raise ValueError('Exp.Ordering must be either a single number or a function handle.')
        n = _nargin(Exp.Ordering)
        if n == 1:
            f1 = Exp.Ordering
            Exp.Ordering = lambda alpha, beta, gamma: f1(beta) * np.ones(np.shape(gamma))
        elif n != 3:
            F.logmsg(1, '  Ordering function in Exp.Ordering must take 1 input argument (beta) or 3 input arguments '
                        '(alpha,beta,gamma).')
    Opt.R_L2S, Opt.rotatedSample = F.p_samplerotation(Exp)[:2]
    if not crystalSample and len(Opt.R_L2S) > 1:
        raise ValueError('For disordered and partially ordered samples, only a single sample orientation '
                         '(Exp.SampleFrame) can be used.')
    Opt.partiallyOrderedSample = partiallyOrderedSample
    Opt.crystalSample = crystalSample
    Opt.disorderedSample = disorderedSample
    return Exp, Opt
