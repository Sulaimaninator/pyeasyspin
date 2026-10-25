"""Frequency/field/g value conversion utility"""
from .._registry import F
from .._struct import Struct



def _c(v):
    return v[0] if isinstance(v, tuple) else v


# eV: EasySpin's GUI uses planck/echarge here, which inverts the conversion; the port uses
# echarge/planck (hv = E) - see DEVIATIONS.md
_FREQ = {'MHz': lambda: 1e6, 'GHz': lambda: 1e9, 'cm^-1': lambda: 100 * _c(F.clight()),
         'eV': lambda: _c(F.echarge()) / _c(F.planck())}
_FIELD = {'mT': 1e-3, 'T': 1.0, 'G': 1e-4, 'kG': 1e-1}


def eprconvert(Freq=None, Field=None, g=None, FreqUnit='GHz', FieldUnit='mT'):
    """Headless version of EasySpin's eprconvert GUI.

    Give two of Freq (in FreqUnit: MHz, GHz, cm^-1, eV), Field (in FieldUnit: mT, T, G, kG) and
    g; the third is computed as the GUI's Compute buttons do (hv = g muB B). With nothing given,
    the GUI start state is used (9.8 GHz, g = gfree) and the field is computed. Returns a Struct
    with FreqSI (Hz), FieldSI (T), g and the GUI's display lists FreqList/FieldList (strings)."""
    if FreqUnit not in _FREQ or FieldUnit not in _FIELD:
        raise ValueError('Unknown unit.')
    FreqSI = None if Freq is None else Freq * _FREQ[FreqUnit]()
    FieldSI = None if Field is None else Field * _FIELD[FieldUnit]
    if FreqSI is None and FieldSI is None and g is None:
        FreqSI, g = 9.8e9, _c(F.gfree())
    if FieldSI is None and g is None:
        g = _c(F.gfree())
    h, muB = _c(F.planck()), _c(F.bmagn())
    if FieldSI is None:
        if FreqSI is None:
            raise ValueError('Invalid frequency value!')
        FieldSI = h * FreqSI / muB / g
    elif g is None:
        if FreqSI is None:
            raise ValueError('Invalid frequency value!')
        g = h * FreqSI / muB / FieldSI
    elif FreqSI is None:
        FreqSI = muB * g * FieldSI / h
    FrqStr = ['%g MHz' % (FreqSI / 1e6), '%g GHz' % (FreqSI / 1e9),
              '%g cm^-1' % (FreqSI / 100 / _c(F.clight())), '%g eV' % (FreqSI * h / _c(F.echarge()))]
    FieldStr = ['%g mT' % (FieldSI * 1e3), '%g T' % FieldSI, '%g G' % (FieldSI * 1e4),
                '%g kG' % (FieldSI * 10)]
    return Struct(FreqSI=FreqSI, FieldSI=FieldSI, g=g, FreqList=FrqStr, FieldList=FieldStr)
