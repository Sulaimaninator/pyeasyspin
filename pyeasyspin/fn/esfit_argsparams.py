"""Parameter bookkeeping for esfit: map fields of {Sys,Exp,Opt}-style argument cells to a
parameter vector and back (private)."""
import copy
import numbers

import numpy as np

from .._registry import nout
from .._struct import Struct
from .._mat import size as msize, numel, isempty


def _isnum(v):
    return isinstance(v, (numbers.Number, np.number)) and not isinstance(v, (bool, np.bool_))


def iscell(v):
    """MATLAB iscell for Python values: a list/tuple is a cell unless it is a plain numeric
    (nested) list such as [2, 2.1] or [[1, 2], [3, 4]], which stands for a numeric array.
    A list holding ndarrays, strings, structs or None is a cell."""
    if not isinstance(v, (list, tuple)):
        return False
    if len(v) == 0:
        return True

    def numeric(x):
        if _isnum(x):
            return True
        if isinstance(x, (list, tuple)) and x:
            return all(numeric(y) for y in x)
        return False
    return not all(numeric(x) for x in v)


def isfloat(v):
    """MATLAB isfloat for Python values (bool and str are not floats; Python ints are doubles)."""
    if _isnum(v):
        return True
    if isinstance(v, np.ndarray):
        return v.dtype.kind in 'fiuc'
    if isinstance(v, (list, tuple)) and not iscell(v):
        return True
    return False


def _arr(value):
    a = np.asarray(value)
    if a.dtype.kind in 'iub':
        a = a.astype(float)
    return a


def _getel(value, i):
    """value(i), i 1-based linear (column-major) index."""
    return _arr(value).reshape(-1, order='F')[i - 1]


def _setel(value, i, newval):
    """value(i) = newval on a copy, i 1-based linear (column-major) index."""
    if _isnum(value) and i == 1:
        return complex(newval) if isinstance(value, complex) else float(newval)
    a = np.array(_arr(value), copy=True)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    if np.iscomplexobj(newval) and not np.iscomplexobj(a):
        a = a.astype(complex)
    a[np.unravel_index(i - 1, a.shape, order='F')] = newval
    return a


@nout(1)
def esfit_argsparams():
    """Returns a Struct of helper functions: getparaminfo, setparamvalues, getparamvalues,
    checkparcompatibility, parinfoequal, validargs."""
    funcs = Struct()
    funcs.getparaminfo = getparaminfo
    funcs.setparamvalues = setparamvalues
    funcs.getparamvalues = getparamvalues
    funcs.checkparcompatibility = checkparcompatibility
    funcs.parinfoequal = parinfoequal
    funcs.validargs = validargs
    return funcs


def getparaminfo(args):
    """List of parameter-info Structs (Arg, Cell, ParCellIndex, FieldName, Index, Subscripts,
    FieldSize, Name), one per float element of every field, rows first. Indices are 1-based."""
    if not iscell(args):
        raise ValueError('Model function arguments are expected as a cell array.')
    parinfo = []
    for a, par in enumerate(args, 1):
        if par is None or (not isinstance(par, dict) and isempty(par)):
            continue
        noCell = not iscell(par)
        if noCell:
            par = [par]
        for c, pc_ in enumerate(par, 1):
            if not isinstance(pc_, dict):
                raise ValueError(f'Arg {a}, cell {c}: struct expected.')
            for fname, FieldValue in pc_.items():
                noParCell = not iscell(FieldValue)
                if noParCell:
                    FieldValue = [FieldValue]
                for pc, value in enumerate(FieldValue, 1):
                    if not isfloat(value):
                        continue
                    siz = tuple(msize(_arr(value)))
                    nr, nc = siz[0], int(np.prod(siz[1:]))
                    n = numel(_arr(value))
                    for ir in range(1, nr + 1):          # reshape(idx,size).' traversal: rows first
                        for ic in range(1, nc + 1):
                            idx = (ic - 1) * nr + ir
                            idxName = '' if n == 1 else f'({ir},{ic})'
                            cellName = '' if noCell else f'{{{c}}}'
                            ParCellName = '' if noParCell else f'{{{pc}}}'
                            parinfo.append(Struct(
                                Arg=a, Cell=0 if noCell else c, ParCellIndex=0 if noParCell else pc,
                                FieldName=fname, Index=idx, Subscripts=[ir, ic], FieldSize=list(siz),
                                Name=f'arg{a}{cellName}.{fname}{ParCellName}{idxName}'))
    return parinfo


def _container(args, p):
    a, c = p.Arg, p.Cell
    return args[a - 1][c - 1] if c > 0 else args[a - 1]


def setparamvalues(args, parinfo, newvals):
    """Copy of args with the parameters in parinfo set to newvals."""
    newvals = np.asarray(newvals).reshape(-1, order='F')
    nParams = len(parinfo)
    if newvals.size != nParams:
        raise ValueError("Number of elements in parameter vector doesn't match number of free parameters.")
    args = copy.deepcopy(list(args))
    for p, pi in enumerate(parinfo):
        s = _container(args, pi)
        f, i, pc = pi.FieldName, pi.Index, pi.ParCellIndex
        if pc > 0:
            fv = list(s[f])
            fv[pc - 1] = _setel(fv[pc - 1], i, newvals[p])
            s[f] = fv
        else:
            s[f] = _setel(s[f], i, newvals[p])
    return args


def getparamvalues(args, parinfo):
    """Parameter vector (1-D) read from args at the positions in parinfo."""
    vals = np.full(len(parinfo), np.nan)
    for p, pi in enumerate(parinfo):
        s = _container(args, pi)
        f, i, pc = pi.FieldName, pi.Index, pi.ParCellIndex
        vals[p] = _getel(s[f][pc - 1] if pc > 0 else s[f], i)
    return vals


def checkparcompatibility(parinfo, args):
    nArgs = len(args)
    for p, pi in enumerate(parinfo, 1):
        a, c, pc, f, i = pi.Arg, pi.Cell, pi.ParCellIndex, pi.FieldName, pi.Index
        if a > nArgs:
            raise ValueError(f'Parameter {p}: Only {nArgs} args given, cannot access arg no {a}.')
        arg_ = args[a - 1]
        if iscell(arg_):
            if c > len(arg_):
                raise ValueError(f'Parameter {p}: Not enough cell entries in argument {a}.')
            arg_ = arg_[c - 1]
        if not isinstance(arg_, dict):
            raise ValueError('Structure expected.')
        if f not in arg_:
            raise ValueError(f'Parameter {p}: Field .{f} is missing in input {a}.')
        N = numel(_arr(arg_[f][pc - 1])) if iscell(arg_[f]) else numel(_arr(arg_[f]))
        if i > N:
            raise ValueError(f"Parameter {p}: Field .{f} doesn't have enough elements - at least {i} expected.")


def parinfoequal(p1, p2):
    nParams = len(p1)
    if len(p2) != nParams:
        raise ValueError('Number of parameters is not equal.')
    err = ''
    for p in range(nParams):
        a, b = p1[p], p2[p]
        if a.Arg != b.Arg:
            err = f'Parameter {p + 1}: arg is different ({a.Arg} vs {b.Arg}).'
        elif a.Cell != b.Cell:
            err = f'Parameter {p + 1}: cell is different ({a.Cell} vs {b.Cell}).'
        elif a.FieldName != b.FieldName:
            err = f"Parameter {p + 1}: field name is different ('{a.FieldName}' vs '{b.FieldName}')."
        elif a.Index != b.Index:
            err = f'Parameter {p + 1}: index is different ({a.Index} vs {b.Index}).'
        if err:
            return err
    return err


def validargs(args):
    if iscell(args):
        for a, args_ in enumerate(args, 1):
            if not iscell(args_) and not isinstance(args_, dict):
                raise ValueError(f'args{{{a}}} must be a cell array or structure array.')
            if not iscell(args_):
                args_ = [args_]
            for c, x in enumerate(args_, 1):
                if not isinstance(x, dict):
                    raise ValueError(f'args{{{a}}}{{{c}}} must be a structure.')
    elif isinstance(args, (list, tuple, np.ndarray)) or _isnum(args):
        if min(msize(_arr(args))) > 1:
            raise ValueError('Parameter vector must be a row or column array.')
    else:
        raise ValueError('args must be a one-dimensional cell array or a numeric array.')
