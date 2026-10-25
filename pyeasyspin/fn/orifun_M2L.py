"""Evaluate an orientation function f(alpha,beta,gamma) (sample-to-molecule Euler angles) for
molecular orientations given as molecule-to-lab angles (phi, theta[, chi])."""
import numpy as np

from .._registry import F
from .._mat import vec
from .._matlab import linspace


def orifun_M2L(f=None, R_L2S=None, phi=None, theta=None, chi=None):
    """Row vector of f values. Without chi, f is integrated over chi (12-point trapezoid, as
    EasySpin)."""
    R_S2L = np.asarray(R_L2S, dtype=float).T
    phi, theta = vec(np.asarray(phi, dtype=float)), vec(np.asarray(theta, dtype=float))
    w = np.zeros(phi.size)
    if chi is None:
        chis = linspace(0, 2 * np.pi, 12)
        dchi = chis[1] - chis[0]
        c, s = np.cos(chis), np.sin(chis)
        for iOri in range(phi.size - 1, -1, -1):
            R_L2M = np.asarray(F.erot(phi[iOri], theta[iOri], 0.0), dtype=float).T.copy()
            xL_M = R_L2M[:, 0:1] * c + R_L2M[:, 1:2] * s
            yL_M = -R_L2M[:, 0:1] * s + R_L2M[:, 1:2] * c
            a, b, g = np.zeros(chis.size), np.zeros(chis.size), np.zeros(chis.size)
            for iChi in range(chis.size - 1, -1, -1):
                R_L2M[:, 0] = xL_M[:, iChi]
                R_L2M[:, 1] = yL_M[:, iChi]
                a[iChi], b[iChi], g[iChi] = (float(np.asarray(x).ravel()[0])
                                             for x in F.eulang(R_L2M @ R_S2L, True, nargout=3))
            ow = vec(np.asarray(f(a, b, g), dtype=float))
            w[iOri] = np.sum((ow[1:] + ow[:-1]) / 2) * dchi     # trapz(ow)*dchi
    else:
        chi = vec(np.asarray(chi, dtype=float))
        for iOri in range(phi.size - 1, -1, -1):
            R_L2M = np.asarray(F.erot(phi[iOri], theta[iOri], chi[iOri]), dtype=float).T
            a, b, g = F.eulang(R_L2M @ R_S2L, True, nargout=3)
            w[iOri] = float(np.asarray(f(a, b, g)).ravel()[0])
    return w.reshape(1, -1)
