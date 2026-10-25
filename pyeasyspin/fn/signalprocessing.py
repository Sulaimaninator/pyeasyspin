"""Signal translation and clean up for spidyan"""
import warnings
import numpy as np
from .._registry import F
from .._struct import Struct
from .._mat import size, vec


def _msize(a):
    return list(size(a))


def signalprocessing(TimeAxis=None, RawSignal=None, FreqTranslations=None):
    """Down-convert simulated detection traces (spidyan): traces with a nonzero frequency
    translation go through rfmixer ('IQdemod' for real, 'IQshift' for complex signals).
    The array shuffling (including MATLAB's reshape-not-permute quirks) follows EasySpin; on any
    error the (reshaped) raw signal is returned with a warning, as in EasySpin."""
    FT = vec(np.asarray(FreqTranslations, dtype=float))
    nDet = FT.size
    isCell = isinstance(RawSignal, list)
    Raw = RawSignal
    SignalSize = None
    try:
        if isCell:
            Processed = [None] * len(Raw)
            nPoints = len(Raw)
        else:
            Raw = np.asarray(Raw)
            SignalSize = _msize(Raw)
            if len(SignalSize) == 2:
                nPoints = 1
                if SignalSize[-1] == nDet:
                    Raw = np.reshape(Raw, [1, 1] + SignalSize, order='F')
                else:
                    Raw = np.reshape(Raw, [1, SignalSize[-2], SignalSize[-1]], order='F')
            else:
                nPoints = int(np.prod(SignalSize[:-2]))
                if SignalSize[-1] == nDet:
                    Raw = np.reshape(Raw, [nPoints, SignalSize[-1], SignalSize[-2]], order='F')
                else:
                    Raw = np.reshape(Raw, [nPoints, SignalSize[-2], SignalSize[-1]], order='F')
            Processed = np.zeros(Raw.shape, dtype=complex)
            TimeAxis = np.asarray(TimeAxis)
            if TimeAxis.ndim > 2:
                TimeAxis = np.reshape(TimeAxis, [nPoints, SignalSize[-1]], order='F')
            elif TimeAxis.ndim < 2:
                TimeAxis = TimeAxis.reshape(1, -1)
        Opt = Struct()
        for iPoint in range(nPoints):
            if isCell:
                Ri = np.asarray(Raw[iPoint])
                Ri = Ri.reshape(1, -1) if Ri.ndim < 2 else Ri
                if Ri.shape[1] == nDet:
                    raise ValueError('reshape of a cell array')   # MATLAB errors here (bug in EasySpin)
                Traces = np.zeros(Ri.shape, dtype=complex)
                DCTimeAxis = vec(np.asarray(TimeAxis[iPoint], dtype=float))
            else:
                DCTimeAxis = TimeAxis[iPoint, :] if TimeAxis.shape[0] > 1 else vec(TimeAxis)
            for iTrace in range(nDet):
                if isCell:
                    RF = Ri[iTrace, :]
                else:
                    RF = Raw[iPoint, iTrace].ravel(order='F')    # squeeze(Raw(iPoint,iTrace,:))
                purelyImag = None
                with np.errstate(divide='ignore', invalid='ignore'):
                    r1 = np.max(np.abs(RF)) / np.max(np.imag(RF))
                    r2 = np.max(np.abs(RF)) / np.max(np.real(RF))
                if r1 > 1e8:
                    RF = np.real(RF)
                    purelyImag = False
                elif r2 > 1e8:
                    RF = np.imag(RF)
                    purelyImag = True
                elif FT[iTrace] != 0:
                    RF = 2 * RF
                diffTime = np.round(np.diff(DCTimeAxis) * 1e10) / 1e10
                dt = np.min(diffTime)
                Opt.dt = dt
                Break = np.concatenate([[1], np.nonzero(diffTime != dt)[0] + 2, [DCTimeAxis.size + 1]])
                if FT[iTrace] != 0:
                    DC = np.zeros(RF.size, dtype=complex)
                    for j in range(Break.size - 1):
                        el = np.arange(Break[j] - 1, Break[j + 1] - 1)
                        tseg = DCTimeAxis[el].reshape(1, -1)
                        rseg = RF[el].reshape(1, -1) if isCell else RF[el].reshape(-1, 1)
                        if not np.iscomplexobj(RF):
                            res = F.rfmixer(tseg, rseg, FT[iTrace], 'IQdemod', Opt)
                            DC[el] = vec(res[1])
                            if purelyImag is None:
                                raise NameError("Unrecognized function or variable 'purelyImag'.")
                            DC = np.real(DC) * 1j if purelyImag else np.real(DC) + 0j
                        else:
                            res = F.rfmixer(tseg, rseg, FT[iTrace], 'IQshift', Opt)
                            DC[el] = vec(res[1])
                    out = DC
                else:
                    out = RF
                if isCell:
                    Traces[iTrace, :] = out
                else:
                    Processed[iPoint, iTrace] = np.reshape(out, Processed[iPoint, iTrace].shape, order='F')
            if isCell:
                Processed[iPoint] = _realify(Traces.T)
        if not isCell:
            Processed = np.reshape(Processed, SignalSize, order='F')
            nd = len(SignalSize)
            Processed = np.transpose(Processed, list(range(nd - 2)) + [nd - 1, nd - 2])
            Processed = _realify(Processed)
    except Exception as EM:
        Processed = Raw
        warnings.warn('Down conversion of the signal was not successful and created an error and the raw '
                      'signal in the simulation frame was returned. The encountered error was: ' + str(EM))
    if not isinstance(Processed, list):
        Processed = np.asarray(Processed)
        nd = max(Processed.ndim, 2)
        if nd == 3 and Processed.shape[0] == 1:
            Processed = Processed.reshape([s for s in Processed.shape if s != 1] or [1, 1])
            if Processed.ndim == 1:
                Processed = Processed.reshape(-1, 1)
        if nd == 2 and size(Processed, 1) == 1 and nDet == 1:
            Processed = Processed.reshape(1, -1).T
    return Processed


def _realify(a):
    a = np.asarray(a)
    return a.real.copy() if np.iscomplexobj(a) and not np.any(a.imag) else a
