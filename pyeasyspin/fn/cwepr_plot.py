"""Plot a cw EPR spectrum (field or frequency sweep), as the simulation functions do without outputs."""
import numpy as np

from .._registry import F
from .._mat import isempty, vec


def cwepr_plot(xAxis=None, spec=None, Exp=None):
    """matplotlib version of EasySpin's cwepr_plot; returns nothing."""
    import matplotlib.pyplot as plt
    x = vec(np.asarray(xAxis, dtype=float))
    spec = np.asarray(spec)
    if bool(Exp.get('FrequencySweep', False)):
        if x[-1] < 1:
            x, xl = x * 1e3, 'frequency (MHz)'
        else:
            xl = 'frequency (GHz)'
        titlestr = '%0.8g mT' % float(np.asarray(Exp.Field).ravel()[0])
    else:
        if x[-1] < 10000:
            xl = 'magnetic field (mT)'
        else:
            x, xl = x * 1e3, 'magnetic field (T)'
        titlestr = '%0.8g GHz' % float(np.asarray(Exp.mwFreq).ravel()[0])
    SR = Exp.get('SampleRotation', None)
    crystalRotation = SR is not None and not isempty(SR) and np.size(SR[1]) > 1
    if crystalRotation and np.atleast_2d(spec).shape[0] == np.size(SR[1]):
        F.stackplot(x, np.atleast_2d(spec), 'maxabs', 2, '')
    else:
        plt.plot(x, np.atleast_2d(spec).T)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel(xl)
    plt.ylabel('intensity (arb.u.)')
    plt.title(titlestr)
