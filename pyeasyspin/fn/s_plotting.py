"""Plotting of spidyan/saffron (thyme mode) signals."""
import numpy as np

from .._registry import F
from .._struct import Struct
from .._mat import vec


def _dimrows(D):
    from .s_sequencer import _dimrows as d
    return d(D)


def s_plotting(TimeAxis=None, Signal=None, Exp=None, Opt=None, draw=True):
    """Prepares (and with draw=True draws, matplotlib) the figures of EasySpin's s_plotting:
    expectation values for single-point detection (printed for one data point, 1-D/2-D plots over
    the indirect dimensions) or transients (one data point, a waterfall/surface over one indirect
    dimension). Returns a Struct with the plotted data: kind, labels, axes and a list of curves."""
    Exp = Struct(Exp)
    Opt = Struct(Opt)
    if 'DetOperator' not in Exp:
        Exp.DetOperator = ['electron coherence']
    DetOperator = Exp.DetOperator if isinstance(Exp.DetOperator, list) else [Exp.DetOperator]
    nDetOps = len(DetOperator)
    Signal = Signal if isinstance(Signal, list) else np.asarray(Signal)
    if 'nPoints' in Exp:
        nPoints = [int(v) for v in vec(np.asarray(Exp.nPoints, dtype=float))]
        nDataPoints = int(np.prod(nPoints))
    else:
        nPoints = []
        nDataPoints = 1
        if not Opt.SinglePointDetection and np.atleast_2d(Signal).shape[0] == nDetOps:
            Signal = np.atleast_2d(Signal).T
    F.logmsg(1, '  setting up figures...')
    LabelsDetectionOp = [None] * nDetOps
    if not Opt.SinglePointDetection or (Opt.SinglePointDetection and nPoints and len(nPoints) < 3 and nDataPoints > 1):
        for k, d in enumerate(DetOperator):
            if isinstance(d, str):
                LabelsDetectionOp[k] = 'signal (arb.u.)' if d == 'electron coherence' else '\\langle' + d + '\\rangle'
            else:
                LabelsDetectionOp[k] = 'expectation value'
    LabelsIndirectDims, AxesIndirectDims = [], []
    for iDim in range(len(nPoints)):
        CurrentDim = 'Dim' + str(iDim + 1)
        name, val = _dimrows(Exp[CurrentDim])[0]
        if np.size(val) == 1:
            AxesIndirectDims.append(float(np.ravel(val)[0]) * np.arange(nPoints[iDim]))
            LabelsIndirectDims.append(CurrentDim + ' / \\Delta' + name)
        else:
            AxesIndirectDims.append(np.arange(1, nPoints[iDim] + 1, dtype=float))
            LabelsIndirectDims.append(CurrentDim + ' / Data Points ' + name)
    if isinstance(TimeAxis, list) and not isinstance(Signal, list):
        TimeAxis = TimeAxis[-1]
    TransientLabel = 'time (\\mus)'
    out = Struct(kind='', curves=[], xlabel='', ylabel='', zlabel='', messages=[])
    if Opt.SinglePointDetection:
        F.logmsg(1, '  single point detection:')
        S = np.atleast_2d(Signal)
        if nDataPoints == 1:
            out.kind = 'values'
            for k, d in enumerate(DetOperator):
                s = ('  Expectation value of ' + d + ':   ') if isinstance(d, str) else \
                    ('  Expectation value of operator no.' + str(k + 1) + ':   ')
                msg = s + str(S[0, k])
                out.messages.append(msg)
                print(msg)
        elif len(nPoints) > 2:
            F.logmsg(1, '  more than two indirect dimensions - stopping')
            out.messages.append('Unable to display more than two indirect dimensions.')
            print(out.messages[-1])
        elif len(nPoints) == 1:
            F.logmsg(1, '  creating plot(s) for one indirect dimension')
            out.kind = 'line'
            A = np.asarray(Signal).reshape(nPoints[0], -1, nDetOps, order='F') if np.size(Signal) else Signal
            for k in range(nDetOps):
                out.curves.append((AxesIndirectDims[0], np.real(A[:, 0, k])))
            out.xlabel, out.ylabel = LabelsIndirectDims[0], LabelsDetectionOp
        else:
            F.logmsg(1, '  creating plot(s) for two indirect dimensions')
            out.kind = 'surface'
            A = np.asarray(Signal).reshape(nPoints[0], nPoints[1], -1, nDetOps, order='F')
            for k in range(nDetOps):
                out.curves.append((AxesIndirectDims[0], AxesIndirectDims[1], np.real(A[:, :, 0, k]).T))
            out.xlabel, out.ylabel = LabelsIndirectDims[0], LabelsIndirectDims[1]
    else:
        if nPoints and len(nPoints) > 1:
            F.logmsg(1, '  more than one indirect dimension in combination with transient - can not plot this - stopping')
            out.messages.append('Unable to display more than two indirect dimensions in combination with transient detection.')
            print(out.messages[-1])
        else:
            F.logmsg(1, '  plotting transient(s)')
            out.xlabel = TransientLabel
            if nDataPoints == 1:
                out.kind = 'transient'
                S = np.atleast_2d(Signal)
                T = vec(TimeAxis)
                if S.shape[0] != T.size:
                    S = S.T
                for k in range(nDetOps):
                    out.curves.append((T, np.real(S[:, k]), np.imag(S[:, k])))
                out.ylabel = LabelsDetectionOp
            else:
                out.kind = 'waterfall'
                out.ylabel, out.zlabel = LabelsIndirectDims[0], LabelsDetectionOp
                if not isinstance(Signal, list):
                    SignalSize = list(np.shape(Signal))
                    if nDetOps == 1:
                        SignalSize.append(1)
                    Signal = np.reshape(Signal, (nDataPoints, SignalSize[-2], SignalSize[-1]), order='F')
                for k in range(nDetOps):
                    if isinstance(Signal, list):
                        for i in range(nDataPoints):
                            T = vec(TimeAxis[i])
                            out.curves.append((T, AxesIndirectDims[0][i] * np.ones(T.size),
                                               np.real(np.asarray(Signal[i]).reshape(T.size, -1, order='F')[:, k])))
                    else:
                        TA = np.atleast_2d(TimeAxis)
                        for i in range(nDataPoints):
                            T = TA[0] if TA.shape[0] == 1 else TA[i]
                            out.curves.append((T, AxesIndirectDims[0][i] * np.ones(T.size), np.real(Signal[i, :, k])))
    if draw and out.kind:
        _draw(out)
    return out


def _draw(out):
    import matplotlib.pyplot as plt
    if out.kind == 'line':
        for (x, y), yl in zip(out.curves, out.ylabel):
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.set_xlabel(out.xlabel)
            ax.set_ylabel(yl)
    elif out.kind == 'surface':
        for x, y, z in out.curves:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            X, Y = np.meshgrid(x, y)
            ax.plot_surface(X, Y, z)
            ax.set_xlabel(out.xlabel)
            ax.set_ylabel(out.ylabel)
            ax.view_init(61.2, 41.2 - 90)
    elif out.kind == 'transient':
        for (t, re, im), yl in zip(out.curves, out.ylabel):
            fig, ax = plt.subplots()
            ax.plot(t, re, t, im)
            ax.set_xlabel(out.xlabel)
            ax.set_ylabel(yl)
    elif out.kind == 'waterfall':
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for t, y, z in out.curves:
            ax.plot(t, y, z)
        ax.set_xlabel(out.xlabel)
        ax.set_ylabel(out.ylabel)
        ax.view_init(61.2, 41.2 - 90)
