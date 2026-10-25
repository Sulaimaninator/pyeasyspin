"""Read data from .d01/.exp SpecMan data files"""
import re
import numpy as np
from .._registry import nout
from .._struct import Struct
from .._mat import reshape
from .._io import BinFile, fgetl_lines, strtok, file_exists


def _squeeze_trailing(shape):
    s = [int(v) for v in shape]
    while len(s) > 2 and s[-1] == 1:
        s.pop()
    return s


@nout(3)
def eprload_specman(FileName=None):
    """(Data, Abscissa, Parameters) from a SpecMan .d01 file and its .exp description."""
    FileName = str(FileName)
    try:
        h = BinFile(FileName, 'ieee-le')
    except OSError:
        raise ValueError("File '" + FileName + "' can not be open for read.")
    ndim1 = int(h.scalar('uint32'))
    dformat = h.scalar('uint32')
    sformat = 'float32' if dformat == 1 else 'double'
    dstrms = []
    ntotal = 1
    for _ in range(ndim1):
        ndim2 = int(h.scalar('int32'))
        dim = h.read(4, 'int32')
        dim[ndim2:] = 1
        total = int(h.scalar('int32'))
        dstrms.append(dict(dim=dim, first=ntotal, total=total))
        ntotal += total
    tmpdat = h.read(ntotal, sformat)

    if ndim1 == 0:
        raise ValueError('No data present')
    elif ndim1 == 2:
        a, b = dstrms
        Data = (tmpdat[a['first'] - 1:a['first'] - 1 + a['total']]
                + 1j * tmpdat[b['first'] - 1:b['first'] - 1 + b['total']])
        Data = reshape(Data, _squeeze_trailing(a['dim']))
    elif ndim1 == 1:
        Data = reshape(tmpdat, _squeeze_trailing(dstrms[0]['dim']))
    else:
        dim = dstrms[0]['dim']
        isthesame = True
        is1D = np.sum(dim != 1) == 1
        for k in range(1, ndim1):
            if np.sum(dim == dstrms[k]['dim']) != 4:
                isthesame = False
                break
            elif is1D and np.sum(dstrms[k]['dim'] != 1) != 1:
                is1D = False
        if isthesame and is1D:
            xdim = int(dim[dim != 1][0])
            Data = reshape(tmpdat, [xdim, ndim1])
        else:
            Data = tmpdat.reshape(-1, 1)
    Data = np.asarray(Data)
    if Data.ndim == 1:
        Data = Data.reshape(-1, 1)

    dscname = FileName.replace('d01', 'exp')
    Parameters = SpecMandsc(dscname)
    ax = SpecManpar(Parameters)
    Abscissa = []
    if 'x' not in ax or np.shape(ax['x'])[0] != Data.shape[0]:
        Abscissa.append(np.arange(1, Data.shape[0] + 1, dtype=float).reshape(1, -1))
    else:
        Abscissa.append(ax['x'])
        Parameters['xlabel'] = ax['xlabel']
    if 'y' in ax:
        ny = Data.shape[1] if Data.ndim > 1 else 1
        if np.shape(ax['y'])[0] != ny:
            Abscissa.append(np.arange(1, ny + 1, dtype=float).reshape(1, -1))
        else:
            Abscissa.append(ax['y'])
            Parameters['ylabel'] = ax['ylabel']
    if len(Abscissa) == 1:
        Abscissa = np.asarray(Abscissa[0]).reshape(-1, 1, order='F')
    return Data, Abscissa, Parameters


def SpecMandsc(filename):
    if not file_exists(filename):
        print('Description file was not found.')
        return Struct()
    par = Struct()
    section = ''
    text = 1
    prg = 1
    for s in fgetl_lines(filename):
        s = s.strip()
        if not s:
            continue
        sect = [i for i, c in enumerate(s) if c in '[]']
        if len(sect) == 2 and sect[0] == 0:
            section = s[1:sect[1]].replace('-', '')
        elif section == 'text':
            par['text' + str(text)] = s
            text += 1
        elif section == 'program':
            par['prg' + str(prg)] = s
            prg += 1
        else:
            a, s = strtok(s, '=')
            a = a.strip()
            a = re.sub(r'[/\x5c ]', '_', a)
            par[section + '_' + a] = s[1:]
    return par


def _str2num(s):
    try:
        v = [float(t) for t in re.split(r'[\s,;]+', s.strip()) if t]
    except ValueError:
        return []
    return v[0] if len(v) == 1 else v


def gettoken(istr, tok):
    pos = istr.find(tok)
    if pos < 0:
        return istr, ''
    return istr[:pos].strip(), istr[pos + len(tok):].strip()


def safeget(strct, fld, deflt):
    return strct[fld] if fld in strct else deflt


def kvgetvalue(s):
    prefix = 'pnumkMGT'
    koeff = [1e-12, 1e-9, 1e-6, 1e-3, 1e3, 1e6, 1e9, 1e12]
    m = next((m for m in re.finditer(r'[-+]?[0-9]*(\.[0-9]+)?', s) if m.end() > m.start()), None)
    if m is None:
        val, unit = float('nan'), ''
    else:
        try:
            val = float(m.group(0))
        except ValueError:
            val = float('nan')
        unit = s[m.end():].strip()
    pref, pref_val = '', 1
    if len(unit) > 1:
        kk = prefix.find(unit[0])
        if kk >= 0:
            val = val * koeff[kk]
            unit = unit[1:]
            pref = prefix[kk]
            pref_val = koeff[kk]
    return val, unit, pref, pref_val


def SpecManpar(par):
    prefix = 'numkMG'
    koeff = [1e-9, 1e-6, 1e-3, 1e3, 1e6, 1e9]
    par = Struct(par)   # MATLAB passes by value: params_trans must not leak into Parameters
    res = Struct()
    res['title'] = safeget(par, 'general_name', '?')
    sweepax = []
    fullfield = 'sweep_transient'
    idx = 0
    triggers = _str2num(safeget(par, 'streams_triggers', '1'))
    while fullfield in par:
        t, rest = strtok(par[fullfield], ',')
        t = t.strip()
        size, rest = strtok(rest[1:], ',')
        size = 1 if t in ('S', 'I', 'A', 'R') else _str2num(size)
        reps, rest = strtok(rest[1:], ',')
        reps = _str2num(reps)
        var = []
        while rest:
            v, rest = strtok(rest[1:], ',')
            var.append(v)
        sweepax.append(dict(t=t, size=size, reps=reps, var=var))
        fullfield = 'sweep_sweep' + str(idx)
        idx += 1
    sweepax[0]['size'] = sweepax[0]['size'] * triggers
    res['sweepax'] = sweepax
    axislabel = 'xyz'
    counter = 0
    for ax in sweepax:
        arr = []
        asize = int(ax['size'])
        if asize <= 1:
            continue
        if ax['t'] in ('I', 'A'):
            tempparam = 'trans'
            par['params_trans'] = '1sl step 1sl;'
        elif ax['t'] == 'T':
            tempparam = 'trans'
            dwell = gettoken(safeget(par, 'streams_dwelltime', '1 ns'), ',')[0].strip()
            par['params_trans'] = '0 ns step ' + dwell + ';'
        else:
            tempparam = ax['var'][0].replace(' ', '_')
        parfield = 'params_' + tempparam
        if parfield in par:
            s = par[parfield]
            if re.search(r'\Wstep\W', s):
                tk1, str1 = gettoken(s, 'step')
                tk2 = gettoken(str1, ';')[0].strip()
                minval, unit = kvgetvalue(tk1)[:2]
                step = kvgetvalue(tk2)[0]
                arr = np.arange(asize) * step + minval
            elif re.search(r'\Wlogto\W', s):
                tk1, str1 = gettoken(s, 'logto')
                tk2 = gettoken(str1, ';')[0].strip()
                minval, unit = kvgetvalue(tk1)[:2]
                maxval = kvgetvalue(tk2)[0]
                arr = np.logspace(np.log10(minval), np.log10(maxval), asize)
            elif re.search(r'\Wto\W', s):
                tk1, str1 = gettoken(s, 'to')
                tk2 = gettoken(str1, ';')[0].strip()
                minval, unit = kvgetvalue(tk1)[:2]
                maxval = kvgetvalue(tk2)[0]
                arr = np.arange(asize) * (1 / (asize - 1)) * (maxval - minval) + minval
            else:
                str1 = gettoken(s, ';')[0]
                tk1, str1 = gettoken(str1, ',')
                vals = []
                while tk1:
                    v, unit = kvgetvalue(tk1)[:2]
                    vals.append(v)
                    tk1, str1 = gettoken(str1, ',')
                    if not tk1 and str1:
                        tk1, str1 = str1, ''
                arr = np.array(vals, dtype=float)
            arr = np.asarray(arr, dtype=float).reshape(1, -1)
        else:
            par['aquisition_' + tempparam]   # MATLAB errors if absent
            arr = np.arange(asize, dtype=float).reshape(-1, 1)
            unit = 's'
        if unit not in ('G', 'K', 's'):
            umax = np.max(np.abs(arr))
            for kk in range(len(koeff) - 1, -1, -1):
                if umax > koeff[kk]:
                    unit = prefix[kk] + unit
                    arr = arr / koeff[kk]
                    break
        res[axislabel[counter]] = arr.T
        res[axislabel[counter] + 'label'] = ax['var'][0] + ', ' + unit
        counter += 1
    return res
