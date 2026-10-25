"""Multicore paths (pyeasyspin/_parallel.py) give the same numbers as the serial code."""
import contextlib
import io

import numpy as np
import pytest

import pyeasyspin as es
from pyeasyspin import _parallel


@pytest.fixture
def force_parallel():
    """Send every batch to the pool (no minimum time), then restore the settings."""
    old = (es.config.Workers, es.config.ParallelMinTime)
    es.config.Workers, es.config.ParallelMinTime = 4, 0.0
    es.parallel_map(es.gfree, [()] * 4)          # start the workers
    yield
    es.config.Workers, es.config.ParallelMinTime = old


def _cu():
    Sys = es.Struct(g=[2.05, 2.05, 2.25], Nucs='63Cu', A=[60, 60, 540], lwpp=1.0)
    Exp = es.Struct(mwFreq=9.5, Range=[250, 365], nPoints=512)
    return Sys, Exp


def test_blas_threads_restored():
    from threadpoolctl import threadpool_info
    es.pepper(*_cu())                     # loads every BLAS library pepper uses
    before = {i['filepath']: i['num_threads'] for i in threadpool_info() if i['user_api'] == 'blas'}
    es.pepper(*_cu())
    after = {i['filepath']: i['num_threads'] for i in threadpool_info() if i['user_api'] == 'blas'}
    assert before == after


def test_parallel_map_order_and_errors(force_parallel):
    Sys, Exp = _cu()
    systems = [es.Struct(Sys, g=[2.05, 2.05, 2.25 + 0.01 * k]) for k in range(6)]
    par = es.parallel_map(es.pepper, [(s, Exp) for s in systems])
    ser = [es.pepper(s, Exp) for s in systems]
    for (xp, yp), (xs, ys) in zip(par, ser):
        assert np.array_equal(xp, xs) and np.array_equal(yp, ys)
    with pytest.raises(Exception, match='mwFreq|Range|field'):
        es.parallel_map(es.pepper, [(Sys, Exp), (Sys, es.Struct(nPoints=512))])


@pytest.mark.filterwarnings('ignore:invalid value encountered in sqrt:RuntimeWarning')   # pstd of a flat direction
@pytest.mark.parametrize('method,opts', [
    ('levmar', {}),
    ('simplex', {}),
    ('swarm', dict(nParticles=8, TolStallIter=2)),
    ('genetic', dict(PopulationSize=8, maxGenerations=2)),
    ('grid', dict(GridSize=2)),
    ('montecarlo', dict(nTrials=10)),
])
def test_esfit_parallel_identical(force_parallel, method, opts):
    # garlic keeps the fits fast; force_parallel sends every batch to the pool regardless
    Sys = es.Struct(g=2.003, Nucs='1H,1H', A=[12.0, 27.0], lwpp=0.06)
    Exp = es.Struct(mwFreq=9.5, Range=[336.5, 341.0], nPoints=512)
    _, data = es.garlic(Sys, Exp)
    data = np.ravel(data) + 0.002 * np.max(np.abs(data)) * np.random.default_rng(1).standard_normal(512)
    Sys0 = es.Struct(g=2.0035, Nucs='1H,1H', A=[11.0, 28.0], lwpp=0.07)
    Vary = es.Struct(g=0.001, A=[3.0, 3.0], lwpp=0.03)

    def fit(workers):
        es.config.Workers = workers
        np.random.seed(20672)
        Opt = es.Struct(Method=method + ' fcn', Verbosity=0, **opts)
        with contextlib.redirect_stdout(io.StringIO()):
            return es.esfit(data, es.garlic, [Sys0, Exp], [Vary], Opt)

    rs, rp = fit(1), fit(4)
    for k in ('pfit', 'rmsd', 'fit', 'pstd'):
        assert np.array_equal(np.asarray(rs[k]), np.asarray(rp[k]), equal_nan=True), k


def _orientations(n, seed=3):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0, 2 * np.pi, n), np.arccos(rng.uniform(-1, 1, n)), rng.uniform(0, 2 * np.pi, n)])


RESFIELDS_CASES = {
    # S = 5/2 with a large D: looping transitions (several resonances per transition), H strain
    's52_looping': (dict(S=2.5, g=2.0, D=3000.0, HStrain=[20, 20, 20]), dict(mwFreq=9.5, Range=[0, 1500])),
    # rhombic Cu(II) with g and A strain, gradients
    'cu_strain': (dict(g=[2.04, 2.06, 2.25], Nucs='63Cu', A=[50, 70, 540], gStrain=[0.01, 0.01, 0.02],
                       AStrain=[0, 0, 30]), dict(mwFreq=9.5, Range=[250, 365])),
    # hybrid (first-order) 14N with tilted A and Q: the perturbation-nucleus arrays
    'n14_hybrid': (dict(S=0.5, g=[2.00, 2.01, 2.02], Nucs='14N', A=[3, 5, 11], AFrame=[0, 0.7, 0],
                        Q=[-1.2, -0.5, 1.7], QFrame=[0.6, 1.1, 0.3]), dict(mwFreq=9.5, Range=[330, 345])),
}


@pytest.mark.parametrize('name', sorted(RESFIELDS_CASES))
def test_resfields_orientation_split_identical(force_parallel, name):
    from pyeasyspin._registry import F
    s, e = RESFIELDS_CASES[name]
    Sys = es.Struct(s)
    Exp = es.Struct(e, SampleFrame=_orientations(20))
    Opt = es.Struct(Method='hybrid') if name == 'n14_hybrid' else es.Struct()
    es.config.Workers = 1
    ser = F.resfields(Sys, Exp, Opt, nargout=5)
    es.config.Workers = 4
    par = F.resfields(Sys, Exp, Opt, nargout=5)
    for a, b in zip(ser, par):
        assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def test_pepper_powder_orientation_split_identical(force_parallel):
    Sys = es.Struct(S=2.5, g=2.0, D=300.0, Nucs='55Mn', A=250.0, lwpp=1.5)
    Exp = es.Struct(mwFreq=9.5, Range=[240, 440], nPoints=2048)
    es.config.Workers = 1
    xs, ys = es.pepper(Sys, Exp)
    es.config.Workers = 4
    xp, yp = es.pepper(Sys, Exp)
    assert np.array_equal(xs, xp) and np.array_equal(ys, yp)


def test_workers_do_not_nest(monkeypatch):
    monkeypatch.setenv('PYEASYSPIN_WORKER', '1')
    assert _parallel.n_workers() == 1
