"""Save data in Bruker BES3T format"""
import datetime
import numpy as np
from .._mat import vec


def _islinear(d):
    d = np.diff(np.asarray(d, dtype=float).ravel(order='F'))
    if d.size == 0:
        return True
    return abs(np.max(d) / np.min(d) - 1) < 1e-5


def eprsave(filename=None, x=None, data=None, TitleString=None, mwFreq=None):
    """Write data (1D or 2D, real or complex) with axis x (2D: [x, y]) to filename.DSC/.DTA
    (BES3T 1.2, big-endian float64). Nonlinear axes go to .XGF/.YGF companion files.
    mwFreq in GHz is stored as MWFQ. Returns nothing."""
    if filename is None:
        raise ValueError('eprsave(filename, x, data, TitleString, mwFreq)')
    if TitleString is None:
        TitleString = ''
    if mwFreq is None:
        mwFreq = np.nan
    data = np.asarray(data)
    if data.ndim > 2:
        raise ValueError('Cannot save data with more than 2 dimensions.')
    if data.ndim < 2:
        data = data.reshape(-1, 1)
    TwoDimData = min(data.shape) > 1
    if TwoDimData:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError('For two-dimensional data, x must be a cell array containing the two axes.')
        y = np.asarray(x[1], dtype=float)
        x = np.asarray(x[0], dtype=float)
    else:
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = np.asarray(x, dtype=float)
    complexData = np.iscomplexobj(data)   # MATLAB ~isreal: complex storage, even if imag is 0

    if len(filename) > 3 and filename[-4:].upper() in ('.DSC', '.DTA'):
        filename = filename[:-4]

    datalist = vec(data)
    if complexData:
        datalist = np.column_stack([datalist.real, datalist.imag]).ravel()   # re/im interleaved
    with open(filename + '.DTA', 'wb') as fh:
        fh.write(np.asarray(datalist, dtype='>f8').tobytes())

    lines = []
    kv = lambda k, v: lines.append(f'{k}\t{v}')
    now = datetime.datetime.now()
    lines.append('* Exported from Matlab using EasySpin, ' + now.strftime('%d-%b-%Y %H:%M:%S'))
    VersionString = '1.2'
    kv('#DESC', VersionString + ' * DESCRIPTOR INFORMATION ***********************')
    kv('DSRC', 'MAN')
    kv('BSEQ', 'BIG')
    kv('IKKF', 'CPLX' if complexData else 'REAL')
    kv('IRFMT', 'D')
    if complexData:
        kv('IIFMT', 'D')
    isLinearX = _islinear(x)
    XType = 'IDX' if isLinearX else 'IGD'
    if TwoDimData:
        isLinearY = _islinear(y)
        YType = 'IDX' if isLinearY else 'IGD'
    else:
        YType = 'NODATA'
    if not isLinearX:
        with open(filename + '.XGF', 'wb') as fh:
            fh.write(np.asarray(vec(x), dtype='>f8').tobytes())
        kv('XFMT', 'D')
    if TwoDimData and not isLinearY:
        with open(filename + '.YGF', 'wb') as fh:
            fh.write(np.asarray(vec(y), dtype='>f8').tobytes())
        kv('YFMT', 'D')
    kv('XTYP', XType)
    kv('YTYP', YType)
    kv('ZTYP', 'NODATA')
    xv = vec(x)
    kv('XPTS', '%d' % xv.size)
    kv('XMIN', '%g' % xv[0])
    kv('XWID', '%g' % (xv[-1] - xv[0]))
    if TwoDimData:
        yv = vec(y)
        kv('YPTS', '%d' % yv.size)
        kv('YMIN', '%g' % yv[0])
        kv('YWID', '%g' % (yv[-1] - yv[0]))
    if TitleString:
        kv('TITL', "'" + TitleString + "'")
    lines += ['*', '************************************************************', '*']
    kv('#SPL', VersionString + ' * STANDARD PARAMETER LAYER')
    kv('OPER', '')
    kv('DATE', now.strftime('%d/%m/%y'))
    kv('TIME', now.strftime('%H:%M:%S'))
    kv('CMNT', '')
    kv('SAMP', '')
    kv('SFOR', '')
    if not np.isnan(mwFreq):
        kv('MWFQ', '%0.9g' % (mwFreq * 1e9))
    with open(filename + '.DSC', 'w', newline='\n', encoding='latin-1') as fh:
        fh.write('\n'.join(lines) + '\n')
