"""Stacked plot of data"""
import numpy as np
from .._registry import F
from .._struct import Struct
from .._mat import isempty


def _colormap(name, n):
    """MATLAB colormap function name(n) as an n x 3 array (parula -> matplotlib's viridis)."""
    import matplotlib
    name = {'parula': 'viridis'}.get(name, name)
    cmap = matplotlib.colormaps[name]
    return np.array([cmap(v)[:3] for v in np.linspace(0, 1, n)]) if n > 1 else np.array([cmap(0.0)[:3]])


def stackplot(x=None, y=None, scale=None, step=None, sliceLabels=None, colors=None, plot=True, ax=None):
    """stackplot(x, y, scale='maxabs', step=1.5, sliceLabels, colors): plot the columns of y
    (slices) stacked on top of each other, each rescaled ('maxabs', 'int', 'dint', 'none', or
    {mode, factor}). step: spacing in units of the largest slice, or a vector of offsets.

    Returns a Struct with x (n x nSlices), yplot (rescaled and shifted slices), shift (offsets,
    also the y tick positions), colors (nSlices x 3) and h (the matplotlib lines, plot=True only;
    MATLAB's single output). The default colours sample viridis instead of MATLAB's parula."""
    nargin = 2
    for k, a in enumerate((scale, step, sliceLabels, colors)):
        if a is not None:
            nargin = 3 + k
    scalefact = None
    if nargin < 3:
        scale = 'maxabs'
        scalefact = 1
    if isinstance(scale, (list, tuple)):
        scalefact = scale[1]
        scale = scale[0]
    if not isinstance(scale, str):
        raise TypeError("Third parameter (scale) must be a string ('maxabs','int','dint' or 'none').")
    if scale == '':
        scale = 'maxabs'
    if scalefact is None:
        scalefact = 1
    if nargin < 4:
        step = np.zeros((0, 0))
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    if y.ndim < 2:
        y = y.reshape(1, -1)
    if y.shape[1] == x.size:
        y = y.T
    nSlices = y.shape[1]
    if nargin < 5:
        sliceLabels = None
    if not isempty(sliceLabels):
        if len(sliceLabels) != nSlices:
            raise ValueError('Number of y tick labels must equal number of slices.')
    if nargin < 6:
        colors = _colormap('parula', nSlices + 1)[:nSlices, :]
    if isinstance(colors, str):
        colors = _colormap(colors, nSlices)
    colors = np.atleast_2d(np.asarray(colors, dtype=float))
    if colors.shape[0] == 1:
        colors = np.tile(colors, (nSlices, 1))
    elif colors.shape[0] != nSlices:
        raise ValueError('Number of colors must equal number of slices.')
    yplot = np.zeros(y.shape, dtype=np.result_type(y, float))
    maxval = np.zeros(nSlices)
    for k in range(nSlices):
        ys = F.rescaledata(y[:, k:k + 1], scale)
        ys = ys[0] if isinstance(ys, tuple) else ys
        yplot[:, k] = np.asarray(ys).ravel() * scalefact
        maxval[k] = np.max(np.abs(yplot[:, k]))
    step = np.asarray(step, dtype=float).ravel()
    if step.size == 0:
        step = np.array([1.5])
    if step.size == 1:
        shift = np.arange(nSlices) * step[0] * np.max(maxval)
    elif step.size == nSlices:
        shift = step
    else:
        raise ValueError('Fourth parameter (step) must be a single number or a vector with'
                         'a number of values equal to the number of slices.')
    if shift[-1] < shift[0]:
        shift = shift[::-1]
    yplot = yplot + shift.reshape(1, -1)
    if min(x.shape) == 1 or x.ndim == 1:
        x = np.tile(x.reshape(-1, 1), (1, y.shape[1]))
    out = Struct(x=x, yplot=yplot, shift=shift, colors=colors, h=None)
    if plot:
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        h = []
        for k in range(nSlices):
            h += ax.plot(x[:, k], np.real(yplot[:, k]), color=colors[k])
        ax.set_yticks(shift)
        ax.set_yticklabels([] if isempty(sliceLabels) else [str(s) for s in sliceLabels])
        ax.autoscale(tight=True)
        out.h = h
    return out
