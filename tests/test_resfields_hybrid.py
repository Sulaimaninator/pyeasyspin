"""resfields with hybrid (first-order) nuclei: pairing of sub-transition intensities and shifts (docs/DEVIATIONS.md).

EasySpin's resfields computes NucTransitionRates = abs(Vu'*Vv).^2, whose rows are u states and
columns v states, then names the row index vidx and takes the shift dEv(vidx) - dEu(uidx). So the
intensity |<u_i|v_j>|^2 is put on the line at dEv(i) - dEu(j) instead of dEv(j) - dEu(i). Its
resfreqs_matrix builds abs(Vf'*Vi).^2 (rows = final states) with the same index names, which is
consistent. The port builds abs(Vv'*Vu).^2 (resfields.EASYSPIN_HYBRID_PAIRING = True restores
EasySpin). Positions and their order are unchanged; for I = 1/2 the 2x2 rate matrix is symmetric
and the intensities are unchanged too. For I >= 1 with nuclear-state mixing EasySpin's intensities
sit on the wrong lines. Reference: full matrix diagonalization (Opt.Method = 'matrix')."""
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import pyeasyspin.fn.resfields as RF
from pyeasyspin._registry import F
from pyeasyspin._struct import Struct

# S = 1/2 + 14N: hyperfine (3-11 MHz, tilted), quadrupole (tilted differently) and 14N Zeeman
# (~1 MHz at 338 mT) compete, so the nuclear eigenstates of the two electron manifolds differ.
SYS = Struct(S=0.5, g=[2.0, 2.01, 2.02], Nucs='14N', A=[3.0, 5.0, 11.0], AFrame=[0.0, 0.7, 0.0],
             Q=[-1.2, -0.5, 1.7], QFrame=[0.6, 1.1, 0.3])
ORIS = np.array([[0, 0, 0], [0.3, 0.5, 0.0], [1.0, 1.0, 0.4], [0.2, 1.4, 0.9], [2.1, 0.8, 0.0]])
EXP = Struct(mwFreq=9.5, Range=[330.0, 350.0], SampleFrame=-np.fliplr(ORIS))


def _resfields(Sys, method):
    P, I = F.resfields(Struct(Sys), Struct(EXP), Struct(Method=method), nargout=2)[:2]
    return np.atleast_2d(P), np.atleast_2d(I)


def _compare(Ph, Ih, Pm, Im, band=None):
    """Worst position difference and worst intensity difference (relative to the strongest line)
    after matching each hybrid line to a matrix line by position, orientation by orientation."""
    dB = dI = 0.0
    for k in range(Ph.shape[1]):
        sel = ~np.isnan(Pm[:, k]) if band is None else (Pm[:, k] > band[0]) & (Pm[:, k] < band[1])
        pm, im = Pm[sel, k], Im[sel, k]
        C = np.abs(Ph[:, k][:, None] - pm[None, :])
        r, c = linear_sum_assignment(C)
        assert r.size == Ph.shape[0] == pm.size
        dB = max(dB, float(np.max(C[r, c])))
        dI = max(dI, float(np.max(np.abs(Ih[r, k] - im[c])) / np.max(im)))
    return dB, dI


@pytest.fixture
def matrix_ref():
    return _resfields(SYS, 'matrix')


def test_hybrid_I1_matches_matrix(matrix_ref, monkeypatch):
    """Corrected pairing: hybrid = matrix up to the first-order approximation (1e-3 of the strongest
    line, positions to 2e-4 mT). EasySpin's pairing is off by up to 25 % of the strongest line."""
    monkeypatch.setattr(RF, 'EASYSPIN_HYBRID_PAIRING', False)
    dB, dI = _compare(*_resfields(SYS, 'hybrid'), *matrix_ref)
    assert dB < 2e-4 and dI < 2e-3
    monkeypatch.setattr(RF, 'EASYSPIN_HYBRID_PAIRING', True)
    dB_es, dI_es = _compare(*_resfields(SYS, 'hybrid'), *matrix_ref)
    assert dB_es < 2e-4 and dI_es > 0.2


def test_easyspin_flag_transposes_pairing(monkeypatch):
    """The flag changes only which intensity sits on which line: positions (and their row order)
    are identical, and per orientation the intensities are the same numbers, permuted."""
    monkeypatch.setattr(RF, 'EASYSPIN_HYBRID_PAIRING', False)
    Pc, Ic = _resfields(SYS, 'hybrid')
    monkeypatch.setattr(RF, 'EASYSPIN_HYBRID_PAIRING', True)
    Pe, Ie = _resfields(SYS, 'hybrid')
    assert np.array_equal(Pc, Pe)
    assert np.allclose(np.sort(Ic, axis=0), np.sort(Ie, axis=0), rtol=1e-12, atol=0)
    assert not np.allclose(Ic, Ie, rtol=1e-3)


def test_hybrid_spin_half_unchanged(monkeypatch):
    """I = 1/2: the 2x2 rate matrix is symmetric, so both pairings give the same output (EasySpin's
    hybrid results with 1H nuclei are unaffected)."""
    Sys = Struct(S=0.5, g=[2.0, 2.01, 2.02], Nucs='1H', A=[8.0, 12.0, 25.0], AFrame=[0.0, 0.7, 0.0])
    out = {}
    for flag in (False, True):
        monkeypatch.setattr(RF, 'EASYSPIN_HYBRID_PAIRING', flag)
        out[flag] = _resfields(Sys, 'hybrid')
    assert np.array_equal(out[False][0], out[True][0])
    assert np.allclose(out[False][1], out[True][1], rtol=1e-12, atol=0)


def test_resfreqs_matrix_hybrid_control():
    """Control: EasySpin's own frequency-domain hybrid branch (consistent pairing, unchanged in the
    port) agrees with its matrix branch at the same first-order level."""
    Exp = Struct(Field=337.5, mwRange=[9.3, 9.7], SampleFrame=-np.fliplr(ORIS))
    Pm, Im = [np.atleast_2d(x) for x in F.resfreqs_matrix(Struct(SYS), Struct(Exp), Struct(Method='matrix'), nargout=2)[:2]]
    Ph, Ih = [np.atleast_2d(x) for x in F.resfreqs_matrix(Struct(SYS), Struct(Exp), Struct(Method='hybrid'), nargout=2)[:2]]
    dnu, dI = _compare(Ph, Ih, Pm, Im, band=(9000.0, 10000.0))
    assert dnu < 1.0 and dI < 2e-3
