"""Bruker BES3T (.DSC/.DTA) loader."""
import os
import warnings
import numpy as np
from .._registry import F, nout
from .._struct import Struct
from .._io import (sscanf_f, num, strtok, isvarname, genvarname, textscan_lines, BinFile)


@nout(3)
def eprload_BrukerBES3T(FullBaseName=None, FileExtension=None, Scaling=None):
    """(Data, Abscissa, Parameters) from FullBaseName + .DSC/.DTA (or lower-case)."""
    if FileExtension in ('.DSC', '.DTA'):
        ParExtension, SpcExtension = '.DSC', '.DTA'
    else:
        ParExtension, SpcExtension = '.dsc', '.dta'
    Parameters, err = readDSCfile(FullBaseName + ParExtension)
    if err:
        raise ValueError(err)

    if 'IKKF' in Parameters:
        parts = Parameters['IKKF'].split(',')
        nDataValues = len(parts)
        isComplex = np.zeros(nDataValues)
        for k in range(nDataValues - 1, -1, -1):
            if parts[k] == 'CPLX':
                isComplex[k] = 1
            elif parts[k] == 'REAL':
                isComplex[k] = 0
            else:
                raise ValueError('Unknown value for keyword IKKF in .DSC file!')
    else:
        warnings.warn('Keyword IKKF not found in .DSC file! Assuming IKKF=REAL.')
        isComplex = np.zeros(1)
        nDataValues = 1

    if 'XPTS' in Parameters:
        nx = num(Parameters['XPTS'])
    else:
        raise ValueError('No XPTS in DSC file.')
    ny = num(Parameters['YPTS']) if 'YPTS' in Parameters else 1
    nz = num(Parameters['ZPTS']) if 'ZPTS' in Parameters else 1
    Dimensions = [int(nx), int(ny), int(nz)]

    if 'BSEQ' in Parameters:
        if Parameters['BSEQ'] == 'BIG':
            ByteOrder = 'ieee-be'
        elif Parameters['BSEQ'] == 'LIT':
            ByteOrder = 'ieee-le'
        else:
            raise ValueError('Unknown value for keyword BSEQ in .DSC file!')
    else:
        warnings.warn('Keyword BSEQ not found in .DSC file! Assuming BSEQ=BIG.')
        ByteOrder = 'ieee-be'

    if 'IRFMT' in Parameters:
        parts = Parameters['IRFMT'].split(',')
        if len(parts) != nDataValues:
            raise ValueError('Problem in BES3T DSC file: inconsistent IKKF and IRFMT fields.')
        fmts = {'C': 'int8', 'S': 'int16', 'I': 'int32', 'F': 'float32', 'D': 'float64'}
        for p in parts:
            p = p.upper()
            if p in fmts:
                NumberFormat = fmts[p]
            elif p == 'A':
                raise ValueError('Cannot read BES3T data in ASCII format!')
            elif p in ('0', 'N'):
                raise ValueError('No BES3T data!')
            else:
                raise ValueError('Unknown value for keyword IRFMT in .DSC file!')
    else:
        raise ValueError('Keyword IRFMT not found in .DSC file!')

    if 'IIFMT' in Parameters:
        a, b = Parameters['IIFMT'].upper(), Parameters['IRFMT'].upper()
        if len(a) != len(b) or a != b:
            raise ValueError('IRFMT and IIFMT in DSC file must be identical.')

    AxisNames = ['X', 'Y', 'Z']
    Abscissa = [None, None, None]
    for a in range(2, -1, -1):
        if Dimensions[a] <= 1:
            continue
        AxisType = Parameters[AxisNames[a] + 'TYP']
        if AxisType == 'IGD':
            companionFileName = FullBaseName + '.' + AxisNames[a] + 'GF'
            DataFormat = Parameters[AxisNames[a] + 'FMT']
            src = {'D': 'float64', 'F': 'float32', 'I': 'int32', 'S': 'int16'}.get(DataFormat)
            if src is None:
                raise ValueError(f'Cannot read data format {DataFormat} for companion file {companionFileName}')
            if os.path.isfile(companionFileName):
                v = BinFile(companionFileName, ByteOrder).read(Dimensions[a], src)
                Abscissa[a] = v.reshape(-1, 1)
                if v.size != Dimensions[a]:
                    raise ValueError(f'Could not read the expected {Dimensions[a]} values from companion file {companionFileName}.')
            else:
                warnings.warn(f'Could not read companion file {companionFileName} for nonlinear axis. Assuming linear axis.')
                AxisType = 'IDX'
        if AxisType == 'IDX':
            Minimum = num(Parameters[AxisNames[a] + 'MIN'])
            Width = num(Parameters[AxisNames[a] + 'WID'])
            if Width == 0:
                print(f'Warning: {AxisNames[a]} range has zero width.')
                Minimum = 1.0
                Width = Dimensions[a] - 1.0
            Abscissa[a] = (Minimum + np.linspace(0, Width, Dimensions[a])).reshape(1, -1)
        if AxisType == 'NTUP':
            raise ValueError('Cannot read data with NTUP axes.')
    # MATLAB cell grows only up to the last assigned axis
    last = max([i for i in range(3) if Abscissa[i] is not None], default=-1)
    Abscissa = [x if x is not None else np.zeros((0, 0)) for x in Abscissa[:last + 1]]
    if len(Abscissa) == 1:
        Abscissa = Abscissa[0].reshape(-1, 1)

    Data = F.getmatrix(FullBaseName + SpcExtension, Dimensions, NumberFormat, ByteOrder, isComplex)

    if Scaling:
        cwExperiment = Parameters['EXPT'] == 'CW'
        DataPreScaled = 'SctNorm' in Parameters and Parameters['SctNorm'].lower() == 'true'
        if 'n' in Scaling:
            if 'AVGS' not in Parameters:
                raise ValueError('Missing AVGS field in the DSC file.')
            nAverages = num(Parameters['AVGS'])
            if DataPreScaled:
                raise ValueError(f'Scaling by number of scans not possible,\nsince data in DSC/DTA are already averaged\nover {int(nAverages)} scans.')
            Data = Data / nAverages
        if cwExperiment and 'G' in Scaling:
            if 'RCAG' not in Parameters:
                raise ValueError('Cannot scale by receiver gain, since RCAG in the DSC file is missing.')
            Data = Data / 10 ** (num(Parameters['RCAG']) / 20)
        if cwExperiment and 'c' in Scaling:
            if 'SPTP' not in Parameters:
                raise ValueError('Cannot scale by sampling time, since SPTP in the DSC file is missing.')
            Data = Data / (num(Parameters['SPTP']) * 1000)
        if cwExperiment:
            if 'P' in Scaling:
                if 'MWPW' not in Parameters:
                    raise ValueError('Cannot scale by power, since MWPW is absent in parameter file.')
                Data = Data / np.sqrt(num(Parameters['MWPW']) * 1000)
        elif 'P' in Scaling:
            raise ValueError('Cannot scale by microwave power, since these are not CW-EPR data.')
        if 'T' in Scaling:
            if 'STMP' not in Parameters:
                raise ValueError('Cannot scale by temperature, since STMP in the DSC file is missing.')
            Data = Data * num(Parameters['STMP'])

    Parameters = F.parsefieldparams(Parameters)
    return Data, Abscissa, Parameters


def readDSCfile(DSCFileName):
    """(Parameters Struct of strings, err message or '')."""
    if not os.path.exists(DSCFileName):
        return Struct(), f'Cannot find the file {DSCFileName}.'
    allLines = textscan_lines(DSCFileName)
    Parameters = Struct()
    k = 0
    while k < len(allLines):
        line = allLines[k]
        if not line:
            k += 1
            continue
        if line[-1] == '\\':
            k2 = k + 1
            while allLines[k2][-1:] == '\\':
                line = line[:-1] + allLines[k2]
                allLines[k2] = ''
                k2 += 1
            line = line[:-1]
            line = line.replace('\\n', '\n')
        Key, Value = strtok(line)
        k += 1
        if not Key:
            continue
        if not Key[0].isalpha():
            if Key.lower() == '#mhl':
                break
            continue
        Value = Value.strip(' \t\n\r\v\f\0')
        if Value and Value[0] == "'" and Value[-1] == "'":
            Value = Value[1:-1] if len(Value) > 1 else ''
        if not isvarname(Key):
            Key = genvarname(Key)
        Parameters[Key] = Value
    return Parameters, ''
