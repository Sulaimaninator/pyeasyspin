"""Resonance fields by the eigenfield method (Belford et al.)."""
import numpy as np
import scipy.linalg

from .._registry import F, nout
from .._struct import Struct
from .._mat import a2, vec, isempty
from ..config import config
from .resfreqs_matrix import _c, _d, _err, _num


@nout(2)
def resfields_eig(Sys=None, Exp=None, Opt=None, nargout=2):
    """(EigenFields, Intensities) in mT; one array per orientation (a list), or the arrays
    themselves for a single orientation. Intensities are computed only for nargout > 1 or a
    nonzero Opt.Threshold; otherwise the second output is empty."""
    if Sys is None or Exp is None:
        raise ValueError('Incorrect number of inputs!')
    if nargout > 2:
        raise ValueError('Incorrect number of outputs.')
    if Opt is None or isempty(Opt):
        Opt = Struct()
    if not isinstance(Sys, dict) or not isinstance(Exp, dict) or not isinstance(Opt, dict):
        raise ValueError('SpinSystem, Parameters and Options must be structures!')
    Opt = Struct(Opt)
    if 'Verbosity' not in Opt:
        Opt.Verbosity = 0.0
    config.LogLevel = Opt.Verbosity

    Sys, err = F.validatespinsys(Sys)
    _err(err)
    if 'Detection' in Exp:
        raise ValueError('Exp.Detection is obsolete. Use Exp.mwMode instead.')
    DefaultExp = Struct(mwFreq=np.nan, Range=np.array([[0.0, np.finfo(float).max]]), mwMode='perpendicular',
                        Temperature=np.nan, SampleFrame=np.zeros((1, 3)), CrystalSymmetry=1.0,
                        MolFrame=np.zeros((1, 3)), SampleRotation=np.zeros((0, 0)))
    Exp = Struct(F.adddefaults(Exp, DefaultExp))
    if np.isnan(_num(Exp.mwFreq)):
        raise ValueError('Parameters.mwFreq missing!')
    Rg = vec(np.asarray(Exp.Range))
    if (Rg.size != 2 or np.iscomplexobj(Rg) or np.any(np.diff(Rg) <= 0) or np.any(~np.isfinite(Rg))
            or np.any(Rg < 0)):
        raise ValueError('Parameters.Range is not valid!')
    Rg = Rg.astype(float)
    if isempty(Exp.mwMode):
        Exp.mwMode = 'perpendicular'
    ParallelMode = _num(_c(F.parseoption(Exp, 'mwMode', ['perpendicular', 'parallel']))) == 2
    T = _num(Exp.Temperature)
    if not np.isnan(T) and not np.isinf(T):
        raise ValueError('Thermal equilibrium populations (Exp.Temperature) not implemented.')
    if 'initState' in Sys and not isempty(Sys.initState):
        raise ValueError('Non-equilibrium populations (Sys.initState) not implemented.')
    mwFreq = _num(Exp.mwFreq) * 1e3

    Orientations, nOrientations, _, averageOverChi = F.p_crystalorientations(Exp, Opt)[:4]
    Orientations = a2(np.asarray(Orientations, dtype=float))
    nOrientations = int(nOrientations)

    Opt = Struct(F.adddefaults(Opt, Struct(Freq2Field=1.0, Threshold=0.0, RejectionRatio=1e-8)))
    if _num(Opt.Freq2Field) not in (0.0, 1.0):
        raise ValueError('Options.Freq2Field incorrect!')
    computeFreq2Field = bool(_num(Opt.Freq2Field))
    th = np.asarray(Opt.Threshold)
    if th.dtype.kind not in 'biuf' or th.size != 1 or float(th) < 0:
        raise ValueError('Opt.Threshold must be a single nonnegative number.')
    Threshold = float(th)
    computeIntensities = nargout > 1 or Threshold != 0
    F.logmsg(1, '  computing %s', 'positions' + (', intensities' if computeIntensities else ''))

    if isinstance(Sys, (list, tuple)):   # unreachable in EasySpin (structure check above)
        H0, mux, muy, muz = Sys
    else:
        H0, mux, muy, muz = F.ham(Sys)[:4]
    H0, mux, muy, muz = (_d(x) for x in (H0, mux, muy, muz))
    n = H0.shape[0]
    A = _d(F.eyekron(H0)) - _d(F.kroneye(np.conj(H0))) + mwFreq * np.eye(n * n)
    # EasySpin: E = diag(eig(A)); all(E>0) runs column-wise over a diagonal matrix, so it is false
    # whenever A is larger than 1x1 and the general eigenproblem is always used (kept as is).
    E = np.diag(np.linalg.eigvals(A))
    SimpleEigenproblem = bool(np.all(np.real(E) > 0))
    F.logmsg(1, '  %s, matrix size %dX%d', 'reduced to simple eigenproblem' if SimpleEigenproblem
             else 'general eigenproblem', A.shape[0], A.shape[0])

    if computeIntensities:
        mux_vec, muy_vec, muz_vec = (m.ravel(order='C') for m in (mux, muy, muz))   # reshape(mu.',[],1)
    RR = float(_num(Opt.RejectionRatio))
    I = np.asarray(Sys.I, dtype=float) if 'I' in Sys and not isempty(Sys.I) else np.zeros(0)
    Polarization = 1.0 / float(np.prod(2 * I + 1))
    EigenFields = [None] * nOrientations
    Intensities = [None] * nOrientations

    for iOri in range(nOrientations):
        F.logmsg(3, '  orientation %d of %d', iOri + 1, nOrientations)
        xLab, yLab, zLab = (vec(x) for x in F.erot(Orientations[iOri, :], 'rows', nargout=3))
        muzL = zLab[0] * mux + zLab[1] * muy + zLab[2] * muz
        B = -_d(F.kroneye(np.conj(muzL))) + _d(F.eyekron(muzL))
        if SimpleEigenproblem:
            BB = np.linalg.solve(A, B)
            w, Vecs = np.linalg.eig(BB)
            with np.errstate(divide='ignore', invalid='ignore'):
                Fields = 1.0 / w
        else:
            Fields, Vecs = scipy.linalg.eig(A, B)
        Fields = np.asarray(Fields)
        with np.errstate(invalid='ignore'):
            keep = ((np.abs(Fields.imag) < RR * np.abs(Fields.real)) & (Fields.real > 0) & np.isfinite(Fields)
                    & (Fields.real > Rg[0]) & (Fields.real < Rg[1]))
        sel = np.flatnonzero(keep)
        sel = sel[np.argsort(Fields.real[sel], kind='stable')]   # MATLAB sort (abs, then angle) on these
        if not computeIntensities:
            EigenFields[iOri] = Fields.real[sel].reshape(-1, 1)
            continue
        if sel.size == 0:
            EigenFields[iOri] = np.zeros((0, 0))
            Intensities[iOri] = np.zeros((0, 0))
            continue
        EF = Fields.real[sel]
        V = Vecs[:, sel]
        V = V / np.sqrt(np.sum(np.abs(V) ** 2, axis=0))
        if ParallelMode:
            muzL_vec = zLab[0] * mux_vec + zLab[1] * muy_vec + zLab[2] * muz_vec
            TransitionRate = np.abs(muzL_vec @ V) ** 2
        else:
            muxL_vec = xLab[0] * mux_vec + xLab[1] * muy_vec + xLab[2] * muz_vec
            if averageOverChi:
                muyL_vec = yLab[0] * mux_vec + yLab[1] * muy_vec + yLab[2] * muz_vec
                TransitionRate = (np.abs(muxL_vec @ V) ** 2 + np.abs(muyL_vec @ V) ** 2) / 2
            else:
                TransitionRate = np.abs(muxL_vec @ V) ** 2
        if computeFreq2Field:
            dBdE = np.empty(V.shape[1])
            for k in range(V.shape[1]):
                Vk = V[:, k].reshape(n, n, order='F')
                dBdE[k] = 1.0 / abs(np.trace(-muzL @ _d(F.commute(Vk, Vk.conj().T))))
        else:
            dBdE = np.ones(TransitionRate.shape)
        Int = Polarization * np.real(TransitionRate * dBdE)
        ok = Int >= Threshold * np.max(Int)
        EigenFields[iOri] = EF[ok].reshape(-1, 1)
        Intensities[iOri] = Int[ok].reshape(-1, 1)

    if not computeIntensities:
        Intensities = [np.zeros((0, 0))]
    if nOrientations == 1:
        EigenFields = EigenFields[0]
        Intensities = Intensities[0]
    return EigenFields, Intensities
