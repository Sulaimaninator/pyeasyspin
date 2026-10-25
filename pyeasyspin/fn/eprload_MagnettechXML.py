"""Magnettech (.xml) loader."""
import base64
import numpy as np
from .._registry import F, nout
from .._struct import Struct
from .._io import sscanf_f


def _as_list(x):
    return x if isinstance(x, list) else [x]


def _interp1(x, y, xi):
    """MATLAB interp1 (linear, NaN outside the range) for monotonic x."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size > 1 and x[0] > x[-1]:
        x, y = x[::-1], y[::-1]
    return np.interp(np.asarray(xi, dtype=float).ravel(), x, y, left=np.nan, right=np.nan)


@nout(3)
def eprload_MagnettechXML(FileName=None):
    """(Data, Abscissa, Parameters) from a Magnettech MS5000 .xml file; Parameters.Curves holds
    every decoded curve (data, x)."""
    Document = F.xml2struct(FileName)
    if 'ESRXmlFile' not in Document:
        raise ValueError(f'File {FileName} is not a Magnettech xml file.')
    E = Document['ESRXmlFile']
    if 'Data' not in E:
        raise ValueError('ESRXmlFile.Data node not found in xml file.')
    if 'Measurement' not in E['Data']:
        raise ValueError('ESRXmlFile.Data.Measurement node not found in xml file.')
    if 'DataCurves' not in E['Data']['Measurement']:
        raise ValueError('ESRXmlFile.Data.Measurement.DataCurves node not found in xml file.')
    if 'Curve' not in E['Data']['Measurement']['DataCurves']:
        raise ValueError('No data. No <Curve> node found in xml file.')
    Measurement = E['Data']['Measurement']
    Parameters = Struct(Measurement['Attributes'])
    CurveList = _as_list(Measurement['DataCurves']['Curve'])

    xAxis = None
    if 'Recipe' in Measurement:
        for p in _as_list(Measurement['Recipe']['Parameters']['Param']):
            Parameters[p['Attributes']['Name']] = p['Text']
        types = {'single': 'field', 'kinetic': 'time', 'temperature': 'field', 'goniometric': 'field',
                 'modulationSweep': 'field', 'powerSweep': 'field', 'xRay': 'field'}
        t = Measurement['Recipe']['Attributes']['Type']
        if t not in types:
            raise ValueError('Unknown Recipe.Type in the xml file.')
        xAxis = types[t]
        if 'BHoldEnabled' in Parameters and Parameters['BHoldEnabled'] == 'True':
            xAxis = 'time'

    Curves = Struct()
    for c in CurveList:
        A = c['Attributes']
        Name = A['YType']
        if Name == 'BField' and A['Mode'] == 'Raw':
            Name = Name + '_' + A['Mode']
        data_txt = c['Text']
        if data_txt:
            if A['Compression'] != 'Base64':
                raise ValueError('Data is not Base64 encoded. Cannot read file.')
            txt = ''.join(data_txt.replace('=', 'A').split())
            bs = bytearray(base64.b64decode(txt + '=' * (-len(txt) % 4)))
            del bs[8::9]
            data = np.frombuffer(bytes(bs[:len(bs) // 8 * 8]), dtype='<f8').astype(float).reshape(-1, 1)
            Curves[Name] = Struct(data=data)
            if data.size != len(data_txt) / 12:
                raise ValueError('Number of base64 chars and number of data points are inconsistent.')
            XOffset = sscanf_f(A['XOffset'])[0][0]
            XSlope = sscanf_f(A['XSlope'])[0][0]
            Curves[Name]['x'] = (XOffset + np.arange(data.size) * XSlope).reshape(1, -1)

    if 'BField' in Curves:
        if xAxis == 'field':
            Abscissa = _interp1(Curves.BField.x, Curves.BField.data, Curves.MW_Absorption.x).reshape(-1, 1)
            Data = Curves.MW_Absorption.data.reshape(-1, 1)
            xmin = float(Parameters['Bfrom'])
            xmax = float(Parameters['Bto'])
            idx = (Abscissa >= xmin) & (Abscissa <= xmax)
            Abscissa = Abscissa[idx].reshape(-1, 1)
            Data = Data[idx].reshape(-1, 1)
        elif xAxis == 'time':
            Abscissa = np.asarray(Curves.MW_Absorption.x).reshape(-1, 1)
            Data = Curves.MW_Absorption.data.reshape(-1, 1)
        else:
            raise ValueError('Cannot construct abscissa for this measurement type')
    elif 'Frequency' in Curves:
        Abscissa = Curves.Frequency.data.reshape(-1, 1)
        Data = Curves.ADC_24bit.data.reshape(-1, 1)
    else:
        raise ValueError('Cannot construct abscissa for this measurement type')
    Parameters = F.parsefieldparams(Parameters)
    Parameters['Curves'] = Curves
    return Data, Abscissa, Parameters
