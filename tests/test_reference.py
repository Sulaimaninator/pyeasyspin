"""pyeasyspin against EasySpin 6.0.12 on standard systems.

The systems are in tests/reference/cases.json. The reference spectra ref_<case>.mat were made
by tests/reference/make_reference.m with EasySpin 6.0.12 in MATLAB. Errors are relative to the
peak of the reference; the tolerance is 1e-6 except for chili (2e-5). chili's default Lanczos
solver runs without reorthogonalization and amplifies rounding: changing g by one part in 1e15
moves its spectrum by about 6e-7 of the peak, so different BLAS libraries differ at ~1e-5.
"""
import json
import os

import numpy as np
import pytest
import scipy.io

import pyeasyspin as es

REF = os.path.join(os.path.dirname(__file__), 'reference')
with open(os.path.join(REF, 'cases.json')) as fh:
    CASES = json.load(fh)


def _struct(d):
    return es.Struct({k: (np.array(v, dtype=float) if isinstance(v, list) else v) for k, v in d.items()})


@pytest.mark.parametrize('name', sorted(CASES))
def test_matches_easyspin(name):
    case = CASES[name]
    ref = scipy.io.loadmat(os.path.join(REF, f'ref_{name}.mat'))
    x_ref, y_ref = ref['x'].ravel(), ref['y'].ravel()
    x, y = getattr(es, case['function'])(_struct(case['Sys']), _struct(case['Exp']))
    x, y = np.ravel(x), np.ravel(y)
    assert x.shape == x_ref.shape and y.shape == y_ref.shape
    np.testing.assert_allclose(x, x_ref, rtol=1e-12, atol=0)
    err = np.max(np.abs(y - y_ref)) / np.max(np.abs(y_ref))
    assert err < case['tol'], f'{name}: max error {err:.2e} of peak (tol {case["tol"]:.0e})'
