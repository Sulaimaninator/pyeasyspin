"""Bruker ESP/WinEPR (.par/.spc) loader."""
import os
import numpy as np
from .._registry import F, nout
from .._struct import Struct
from .._io import num, isvarname, genvarname, textscan_lines


def _f(P, key):
    """sscanf(P.key,'%f') or [] (None) if the field is absent."""
    if key not in P:
        return None
    v = num(P[key])
    return None if isinstance(v, np.ndarray) and v.size == 0 else v


@nout(3)
def eprload_BrukerESP(FullBaseName=None, FileExtension=None, Scaling=None):
    """(Data, Abscissa, Parameters) from FullBaseName + .par/.spc (upper case if given so)."""
    ParExtension, SpcExtension = '.par', '.spc'
    if FileExtension in ('.PAR', '.SPC'):
        ParExtension, SpcExtension = '.PAR', '.SPC'
    Parameters, err = eprload_readPARfile(FullBaseName + ParExtension)
    if err:
        raise ValueError(err)

    FileType = 'c'
    TwoD = False
    isComplex = 0
    nx, ny = 1024, 1
    if 'DOS' in Parameters:
        Endian, FileType = 'ieee-le', 'w'
    else:
        Endian = 'ieee-be'
    if 'JSS' in Parameters:
        Flags = int(num(Parameters['JSS']))
        isComplex = (Flags >> 4) & 1
        TwoD = bool((Flags >> 12) & 1)
    if 'SSX' in Parameters and TwoD:
        if FileType == 'c':
            FileType = 'p'
        nx = num(Parameters['SSX'])
        if isComplex:
            nx = nx / 2
    if 'SSY' in Parameters and TwoD:
        if FileType == 'c':
            FileType = 'p'
        ny = num(Parameters['SSY'])
    if 'ANZ' in Parameters:
        nAnz = num(Parameters['ANZ'])
        if not TwoD:
            if FileType == 'c':
                FileType = 'p'
            nx = nAnz
            if isComplex:
                nx = nx / 2
        elif nx * ny != nAnz:
            raise ValueError('Two-dimensional data: SSX, SSY and ANZ in .par file are inconsistent.')
    if 'RES' in Parameters:
        nx = num(Parameters['RES'])
    if 'REY' in Parameters:
        ny = num(Parameters['REY'])
    if 'XPLS' in Parameters:
        nx = num(Parameters['XPLS'])
    NumberFormat = {'c': 'int32', 'w': 'float', 'p': 'int32'}[FileType]
    nx, ny = int(nx), int(ny)

    Abscissa = np.zeros((0, 0))
    JEY_PowerSweep = False
    XYLB = XYWI = None
    if nx > 1:
        if 'JEX' not in Parameters:
            Parameters['JEX'] = 'field-sweep'
        if 'JEY' not in Parameters:
            Parameters['JEY'] = ''
        JEX_Endor = Parameters['JEX'] == 'ENDOR'
        JEX_TimeSweep = Parameters['JEX'] == 'Time-Sweep'
        JEY_PowerSweep = Parameters['JEY'] == 'mw-power-sweep'
        HCF, HSW, GST, GSI = (_f(Parameters, k) for k in ('HCF', 'HSW', 'GST', 'GSI'))
        XXLB, XXWI, XYLB, XYWI = (_f(Parameters, k) for k in ('XXLB', 'XXWI', 'XYLB', 'XYWI'))
        TakeGH = 0
        if JEX_Endor:
            TakeGH = 1
        elif None not in (XXLB, XXWI, XYLB, XYWI):
            TakeGH = 3
        elif None not in (HCF, HSW, GST, GSI):
            TakeGH = 1
        elif None not in (HCF, HSW):
            TakeGH = 2
        elif None not in (GST, GSI):
            TakeGH = 1
        elif GSI is None and HSW is None:
            HSW = 50
            TakeGH = 2
        elif HCF is None:
            TakeGH = 3
        if JEX_TimeSweep:
            ConversionTime = num(Parameters['RCT']) if 'RCT' in Parameters else 1
            Abscissa = (np.arange(nx) * ConversionTime / 1e3).reshape(1, -1)
        else:
            Abscissa = np.zeros((0, 0))
            if TakeGH == 1:
                Abscissa = (GST + GSI * np.linspace(0, 1, nx)).reshape(1, -1)
            elif TakeGH == 2:
                Abscissa = (HCF + HSW / 2 * np.linspace(-1, 1, nx)).reshape(1, -1)
            elif TakeGH == 3:
                if XXLB is not None and XXWI is not None:
                    if XYLB is not None and XYWI is not None:
                        Abscissa = [(XXLB + np.linspace(0, XXWI, nx)).reshape(1, -1),
                                    (XYLB + np.linspace(0, XYWI, ny)).reshape(1, -1)]
                    else:
                        Abscissa = (XXLB + np.linspace(0, XXWI, nx)).reshape(1, -1)
            else:
                raise ValueError('Could not determine abscissa range from parameter file!')

    if not TwoD and ny > 1:
        ny = 1
    Dimensions = [nx, ny, 1]
    Data = F.getmatrix(FullBaseName + SpcExtension, Dimensions, NumberFormat, Endian, isComplex)
    if ny == 1:
        Data = np.asarray(Data).reshape(1, -1, order='F')

    if Scaling:
        if 'n' in Scaling:
            if 'JSD' not in Parameters:
                raise ValueError('Cannot scale by number of scans, since JSD is absent in parameter file.')
            Data = Data / num(Parameters['JSD'])
        if 'G' in Scaling:
            if 'RRG' not in Parameters:
                raise ValueError('Cannot scale by gain, since RRG is absent in parameter file.')
            Data = Data / num(Parameters['RRG'])
        if 'P' in Scaling:
            if 'MP' not in Parameters:
                raise ValueError('Cannot scale by power, since MP is absent in parameter file.')
            if not JEY_PowerSweep:
                Data = Data / np.sqrt(num(Parameters['MP']))
            else:
                Data = np.array(Data, dtype=float if np.isrealobj(Data) else complex)
                nPowers = Data.shape[1]
                dB = XYLB + np.linspace(0, XYWI, nPowers)
                mwPower = num(Parameters['MP']) * 10.0 ** (-dB / 10)
                for i in range(nPowers):
                    Data[:, i] = Data[:, i] / np.sqrt(mwPower[i])
        if 'T' in Scaling:
            if 'TE' not in Parameters:
                raise ValueError('Cannot scale by temperature, since TE is absent in parameter file.')
            Temperature = num(Parameters['TE'])
            if Temperature == 0:
                raise ValueError('Cannot scale by temperature, since TE is zero in parameter file.')
            Data = Data * Temperature
        if 'c' in Scaling:
            if 'RCT' not in Parameters:
                raise ValueError('Cannot scale by sampling time, since RCT in the .par file is missing.')
            Data = Data / num(Parameters['RCT'])

    Parameters = F.parsefieldparams(Parameters)
    return Data, Abscissa, Parameters


def eprload_readPARfile(PARFileName):
    """(Parameters Struct of strings, err message or '')."""
    if not os.path.exists(PARFileName):
        return Struct(), f'Cannot find the file {PARFileName}.'
    Parameters = Struct()
    for line in textscan_lines(PARFileName):
        if not line:
            continue
        s = line.lstrip(' \t\n\r\v\f')
        if not s:
            continue
        i = len(line) - len(s)
        j = i
        while j < len(line) and line[j] not in ' \t\n\r\v\f':
            j += 1
        Key = line[i:j]
        if not Key[0].isalpha():
            continue
        Value = line[j:].strip(' \t\n\r\v\f\0')
        if Value and Value[0] == "'" and Value[-1] == "'":
            Value = Value[1:-1] if len(Value) > 1 else ''
        if not isvarname(Key):
            Key = genvarname(Key)
        Parameters[Key] = Value
    return Parameters, ''
