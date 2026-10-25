"""Unit conversion tool"""
import numpy as np
from .._registry import F

_UNITS = ['cm^-1->eV', 'cm^-1->K', 'cm^-1->mT', 'cm^-1->MHz', 'eV->cm^-1', 'eV->K', 'eV->mT', 'eV->MHz',
          'K->cm^-1', 'K->eV', 'K->mT', 'K->MHz', 'mT->cm^-1', 'mT->eV', 'mT->K', 'mT->MHz',
          'MHz->cm^-1', 'MHz->eV', 'MHz->K', 'MHz->mT']


def _c(name):
    v = getattr(F, name)()
    return v[0] if isinstance(v, tuple) else v


def unitconvert(value=None, units=None, g=None):
    """unitconvert(value, 'A->B', g=gfree) with A, B in cm^-1, eV, K, mT, MHz.
    The constants are combined in EasySpin's evaluation order (bit-level parity)."""
    if value is None or units is None:
        raise ValueError('Wrong number of input arguments')
    if g is None:
        g = _c('gfree')
    if units not in _UNITS:
        lows = [u for u in _UNITS if u.lower() == str(units).lower()]
        if lows:
            raise ValueError('You provided: %s. Did you mean %s?' % (units, lows[0]))
        raise ValueError('Unknown unit conversion specified.')
    v = value if np.isscalar(value) else np.asarray(value)
    if not np.isscalar(g):
        g = np.asarray(g)
    clight, planck, evolt, boltzm, bmagn = (_c(n) for n in ('clight', 'planck', 'evolt', 'boltzm', 'bmagn'))
    fn = {
        'cm^-1->eV': lambda v: v * 100 * clight * planck / evolt,
        'cm^-1->K': lambda v: v * 100 * clight * planck / boltzm,
        'cm^-1->mT': lambda v: v / g * (planck / bmagn / 1e-3) * 100 * clight,
        'cm^-1->MHz': lambda v: v * 100 * clight / 1e6,
        'eV->cm^-1': lambda v: v * evolt / 100 / clight / planck,
        'eV->K': lambda v: v * evolt / boltzm,
        'eV->mT': lambda v: v / g / bmagn / 1e-3 * evolt,
        'eV->MHz': lambda v: v * evolt / planck / 1e6,
        'K->cm^-1': lambda v: v * boltzm / 100 / clight / planck,
        'K->eV': lambda v: v * boltzm / evolt,
        'K->mT': lambda v: v / g / bmagn / 1e-3 * boltzm,
        'K->MHz': lambda v: v * boltzm / planck / 1e6,
        'mT->cm^-1': lambda v: v * g / (planck / bmagn / 1e-3) / 100 / clight,
        'mT->eV': lambda v: v * g * bmagn * 1e-3 / evolt,
        'mT->K': lambda v: v * g * bmagn * 1e-3 / boltzm,
        'mT->MHz': lambda v: v * g * (1e-3 * bmagn / planck / 1e6),
        'MHz->cm^-1': lambda v: v * 1e6 / 100 / clight,
        'MHz->eV': lambda v: v * 1e6 * planck / evolt,
        'MHz->K': lambda v: v * 1e6 * planck / boltzm,
        'MHz->mT': lambda v: v / g * (planck / bmagn / 1e-3) * 1e6,
    }[units]
    return fn(v)
