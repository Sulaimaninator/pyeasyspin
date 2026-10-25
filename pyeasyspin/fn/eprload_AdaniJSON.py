"""Adani e-Spinoza (.json) loader."""
import json
import numpy as np
from .._registry import nout
from .._struct import Struct


def _fromjson(v):
    """JSON value -> MATLAB value as matlab.internal.webservices.fromJSON builds it: objects ->
    Struct, arrays of numbers/bools -> column vectors, arrays of objects with equal keys -> struct
    array (one element -> Struct, several -> list), other arrays -> cell (list), null -> []."""
    if isinstance(v, dict):
        return Struct((k, _fromjson(x)) for k, x in v.items())
    if v is None:
        return np.zeros((0, 0))
    if isinstance(v, list):
        if not v:
            return np.zeros((0, 0))
        if all(isinstance(x, bool) for x in v):
            return np.array(v, dtype=bool).reshape(-1, 1)
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
            return np.array(v, dtype=float).reshape(-1, 1)
        if all(isinstance(x, dict) for x in v) and all(list(x) == list(v[0]) for x in v):
            items = [_fromjson(x) for x in v]
            return items[0] if len(items) == 1 else items
        return [_fromjson(x) for x in v]
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return float(v)
    return v


def _el(s, k):
    """k-th element (0-based) of a struct array that may be a single Struct."""
    return s if isinstance(s, dict) else s[k]


def _numel(s):
    return 1 if isinstance(s, dict) else len(s)


@nout(3)
def eprload_AdaniJSON(FileName=None):
    """(Data, Abscissa, Parameters) from an Adani JSON file; Parameters is the whole file."""
    try:
        with open(FileName, 'r', encoding='utf-8-sig') as fh:
            raw = fh.read()
    except OSError:
        raise ValueError(f'Could not open {FileName}.')
    try:
        data = _fromjson(json.loads(raw))
    except ValueError:
        raise ValueError(f'Could not parse JSON-format data from {FileName}.')
    nSpectra = _numel(data.Values)
    nPoints = _numel(_el(data.Values, 0).Values)
    nPhases = np.asarray(_el(_el(data.Values, 0).Values, 0).Points).size
    phase0 = data.Phase if 'Phase' in data else 0
    s = np.sin(2 * np.pi * np.arange(nPhases) / nPhases + phase0)
    spc = np.zeros((nPoints, nSpectra))
    for iS in range(nSpectra):
        acq = _el(data.Values, iS).Values
        for iP in range(nPoints):
            spc[iP, iS] = np.sum(np.asarray(_el(acq, iP).Points, dtype=float).ravel() * s)
    d_ = data.ExperimentOptions
    CenterField = d_.CommonOptions.CenterMagneticField
    SweepWidth = d_.CommonOptions.SweepWidth
    B = (np.linspace(-1, 1, nPoints) * SweepWidth / 2 + CenterField).reshape(1, -1)
    if nSpectra == 1:
        Abscissa = B
    else:
        Abscissa = [B, np.linspace(d_.InitialValue2D, d_.FinalValue2D, nSpectra).reshape(1, -1)]
    return spc, Abscissa, data
