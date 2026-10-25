"""Checks of functions and branches that EasySpin's own tests do not reach: Fresnel integrals, DEER kernel,
LPSVD fits, HYSCORE outline, resonator with a measured transfer function, pulse branches, and the
MATLAB-builtin helpers (colon, fminsearch) the ports rely on."""
import math
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import fresnel
from pyeasyspin._registry import F
from pyeasyspin._struct import Struct


def test_fresnel_vs_scipy():
    x = np.linspace(-12, 12, 4001)
    S, C = fresnel(x)
    assert np.max(np.abs(F.fresnelC(x) - C)) < 1e-13
    assert np.max(np.abs(F.fresnelS(x) - S)) < 1e-13
    s, c = fresnel(0.7)
    assert abs(F.fresnelC(0.7j) - 1j * c) < 1e-14 and abs(F.fresnelS(0.7j) + 1j * s) < 1e-14
    assert abs(F.fresnelC(2.0, 1) - fresnel(math.sqrt(2 / math.pi) * 2)[1]) < 1e-13
    assert abs(F.fresnelS(2.0, 2) - fresnel(math.sqrt(4 / math.pi))[0]) < 1e-13


def test_dipkernel_vs_quadrature():
    t = np.linspace(0, 3, 16)
    r = np.array([2.0, 3.0, 4.0])
    K = F.dipkernel(t, r) / (r[1] - r[0])
    D = (F.mu0()[0] / 4 / np.pi) * F.bmagn()[0] ** 2 * F.gfree()[0] ** 2 / F.planck() * 1e21
    Kq = np.array([[quad(lambda z: np.cos((1 - 3 * z * z) * 2 * np.pi * D / rr ** 3 * tt), 0, 1,
                         epsabs=1e-14, epsrel=1e-13, limit=200)[0] for rr in r] for tt in t])
    assert K.shape == (16, 3)
    assert np.max(np.abs(K - Kq)) < 1e-12
    assert np.allclose(F.dipkernel(t, 3.0)[:, 0], Kq[:, 1], atol=1e-12)   # single r: no dr factor
    with pytest.raises(ValueError):
        F.dipkernel(t, np.array([2.0, 3.0, 5.0]))


def _sig(t):
    return (2.0 * np.exp(-0.8 * t) * np.exp(1j * (2 * np.pi * 3.0 * t + 0.3))
            + 0.7 * np.exp(-1.5 * t) * np.exp(1j * (2 * np.pi * -7.0 * t + 1.1)))


@pytest.mark.parametrize('method', ['kt', 'ss', 'tls'])
def test_lpsvd_recovers_two_components(method):
    t = np.arange(256) * 0.01
    y, p = F.lpsvd(_sig(t), t, method, 2)
    assert np.max(np.abs(y - _sig(t))) < 1e-10
    assert np.allclose(p.frequency.ravel(), [-7, 3], atol=1e-9)
    assert np.allclose(p.damping.ravel(), [1.5, 0.8], atol=1e-9)
    assert np.allclose(p.amplitude.ravel(), [0.7, 2.0], atol=1e-9)
    assert np.allclose(p.phase.ravel(), [1.1, 0.3], atol=1e-9)
    assert np.max(np.abs(p.model(t).ravel() - _sig(t))) < 1e-10


def test_mlpsvd_sum_and_orientation():
    t = np.arange(256) * 0.01
    data = np.vstack([_sig(t), 0.5 * np.exp(0.2j) * _sig(t), -0.3 * _sig(t)])
    for method in ('kt', 'tls'):
        y, p = F.mlpsvd(data, t, method, 'mdl', 'sum')
        assert np.max(np.abs(y - data)) < 1e-10
        assert np.allclose(p.frequency.ravel(), [-7, 3], atol=1e-9)
    y, p = F.mlpsvd(data.T, t, 'ss', 2)
    assert y.shape == (256, 3) and np.max(np.abs(y - data.T)) < 1e-10


def test_nucfrq2d_isotropic_hyperfine():
    Sys = Struct(S=0.5, g=2.0, Nucs='1H', A=6.0)
    o = F.nucfrq2d(Sys, 350.0)
    nuI = o.LarmorFreqs[0]
    # beta manifold (mS = -1/2, lower energies) = nuI + A/2; alpha = |nuI - A/2|; 2nd-order shifts ~1e-3
    assert abs(o.FreqsBeta[0, 0] - (nuI + 3)) < 2e-3
    assert abs(o.FreqsAlpha[0, 0] - abs(nuI - 3)) < 2e-3
    o2 = F.nucfrq2d(Struct(S=0.5, g=2.0, Nucs='1H', A=np.array([[4.0, 4.0, 10.0]])), 350.0, 0.12)
    assert o2.Modulation.shape == (201, 201) and abs(np.max(o2.Modulation) - 1) < 1e-15


def test_resonator_measured_transfer_function_matches_ideal():
    t, IQ = F.pulse(Struct(tp=0.06, Type='rectangular', Flip=np.pi, TimeStep=0.0005))
    ti, si = F.resonator(t, IQ, 9.5, 9.5, 200.0, 'simulate')
    nu = np.linspace(9.2, 9.8, 301)
    H = F.resonatorprofile(nu, 9.5, 200.0, 'transferfunction')
    for HH in (H, np.abs(H)):        # complex response, and magnitude only (phase via Hilbert transform)
        te, se = F.resonator(t, IQ, 9.5, nu, HH, 'simulate')
        n = min(si.size, se.size)
        assert np.max(np.abs(si.ravel()[:n] - se.ravel()[:n])) < 0.005 * np.max(np.abs(si))


def test_pulse_bw_compensation_frequencyresponse_matches_ideal():
    base = dict(tp=0.128, TimeStep=1e-5, Type='quartersin/linear', trise=0.03, Frequency=np.array([-150, 150]),
                Amplitude=1.0, mwFreq=9.5)
    _, I1 = F.pulse(Struct(base, ResonatorFrequency=9.5, ResonatorQL=50))
    nu = np.linspace(9.0, 10.0, 2001)
    H = F.resonatorprofile(nu, 9.5, 50.0, 'transferfunction')
    _, I2 = F.pulse(Struct(base, FrequencyResponse=np.vstack([nu, np.abs(H)])))
    _, I3 = F.pulse(Struct(base, FrequencyResponse=np.vstack([nu, np.abs(H)]).T))
    assert np.max(np.abs(I1 - I2)) < 1e-4 and np.array_equal(I2, I3)


def test_pulse_unrecorded_shapes_and_databases():
    t, IQ = F.pulse(Struct(tp=0.1, Type='halfsin', Flip=np.pi / 2))
    assert abs(2 * np.pi * np.trapezoid(np.real(IQ).ravel(), t.ravel()) - np.pi / 2) < 1e-12
    t, IQ = F.pulse(Struct(tp=0.2, Type='Q5', Amplitude=1))     # database entry, normalized to max 1
    assert abs(np.max(IQ) - 1) < 1e-15
    from pyeasyspin.fn.pulse import _read_database, _GAUSSIAN_CASCADE_TXT, _FOURIER_SERIES_TXT
    d = _read_database(_GAUSSIAN_CASCADE_TXT, 'q5', 6, ('x0', 'A0', 'FWHM'))
    assert np.allclose(d['A0'].ravel(), [-1.48, -4.34, 7.33, -2.30, 5.66])
    d = _read_database(_FOURIER_SERIES_TXT, 'snob i3', 5, ('A0', 'An', 'Bn'))
    assert d['A0'].shape == (1, 1) and d['An'].size == 9 and d['Bn'][0, 0] == -1.1990
    with pytest.raises(ValueError, match="'foo' is not defined"):
        F.pulse(Struct(tp=0.1, Type='foo', Flip=np.pi))


def test_matlab_colon_and_fminsearch():
    from pyeasyspin.fn.pulse import _colon
    from pyeasyspin.fn.resonator import _fminsearch
    x = _colon(0, 0.1, 1)
    assert x.size == 11 and x[-1] == 1.0 and x[5] == 0.5
    assert _colon(0, 3.8e-06, 0.1).size == 26316
    assert _colon(1, -1, 2).size == 0
    n = [0]

    def rosen(v):
        n[0] += 1
        return 100 * (v[1] - v[0] ** 2) ** 2 + (1 - v[0]) ** 2
    v = _fminsearch(rosen, [-1.2, 1])
    # MATLAB documentation example: fval 8.1777e-10 after 159 function evaluations
    assert abs(rosen(v) - 8.1777e-10) < 1e-13 and n[0] - 1 == 159
