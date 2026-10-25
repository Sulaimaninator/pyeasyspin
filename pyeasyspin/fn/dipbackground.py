"""Intermolecular dipolar background decay"""
import math
import numpy as np
from .._registry import F


def _v(x):
    return x[0] if isinstance(x, tuple) else x


def dipbackground(t=None, conc_uM=None, lambda_=None, plot=False):
    """dipbackground(t, conc_uM, lambda) -> Vinter: homogeneous 3D background decay for time t (µs),
    spin concentration (µM) and modulation depth lambda. plot=True draws it (MATLAB nargout=0)."""
    if t is None:
        return None
    g1 = _v(F.gfree())
    g2 = _v(F.gfree())
    D = _v(F.mu0()) / (4 * math.pi) * g1 * g2 * _v(F.bmagn()) ** 2 / _v(F.hbar())
    conc_mM = conc_uM / 1e3
    conc = conc_mM * _v(F.avogadro())
    t = np.asarray(t, dtype=float) * 1e-6
    k = 8 * math.pi ** 2 / (9 * math.sqrt(3)) * D * conc * lambda_
    Vinter = np.exp(-k * np.abs(t))
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(np.ravel(t) / 1e-6, np.ravel(Vinter))
        plt.xlabel('time (µs)')
        plt.ylabel('V_inter')
        plt.grid(True)
        plt.show()
    return Vinter
