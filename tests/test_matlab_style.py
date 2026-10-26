"""MATLAB-style use of the public namespace: star import, and vectors returned as 1-D arrays."""
import numpy as np
import pytest

import pyeasyspin as es


def _nitroxide():
    Sys = es.Struct()
    Sys.g = [2.0088, 2.0061, 2.0027]
    Sys.Nucs = '14N'
    Sys.A = [16, 16, 86]
    Sys.lwpp = [0, 0.3]
    Exp = es.Struct()
    Exp.mwFreq = 9.5
    Exp.Range = [328, 349]
    return Sys, Exp


def test_star_import_gives_every_public_function():
    ns = {}
    exec('from pyeasyspin import *', ns)
    for name in ('Struct', 'config', 'parallel_map', 'pepper', 'garlic', 'chili', 'esfit', 'eprload'):
        assert name in ns
    assert set(es.__all__) <= set(ns)
    Sys, Exp = _nitroxide()
    B, spc = ns['pepper'](Sys, Exp)          # pepper(Sys, Exp), as in MATLAB
    assert B.shape == spc.shape == (1024,)


def test_vectors_are_1d_and_matrices_keep_their_shape():
    Sys, Exp = _nitroxide()
    B, spc = es.pepper(Sys, Exp)
    assert B.ndim == 1 and spc.ndim == 1
    dspc = es.deriv(B, spc)                   # outputs go back in as inputs
    assert dspc.shape == spc.shape
    _, comps = es.pepper([Sys, es.Struct(g=2.05, lwpp=1.0, weight=0.3)], Exp, es.Struct(separate='components'))
    assert comps.shape == (2, 1024)


def test_matlab_shapes_switch():
    Sys, Exp = _nitroxide()
    B, spc = es.pepper(Sys, Exp)
    es.config.MatlabShapes = True
    try:
        B2, spc2 = es.pepper(Sys, Exp)
    finally:
        es.config.MatlabShapes = False
    assert B2.shape == spc2.shape == (1, 1024)
    assert np.array_equal(B2.ravel(), B) and np.array_equal(spc2.ravel(), spc)


def test_plot_draws_one_line():
    mpl = pytest.importorskip('matplotlib')
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    B, spc = es.pepper(*_nitroxide())
    lines = plt.plot(B, spc)
    plt.close('all')
    assert len(lines) == 1
