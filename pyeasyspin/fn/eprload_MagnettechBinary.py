"""Magnettech binary (.spe) loader."""
import os
import numpy as np
from .._registry import nout
from .._struct import Struct
from .._io import BinFile


@nout(3)
def eprload_MagnettechBinary(FileName=None):
    """(Data, Abscissa, Parameters) from a Magnettech MiniScope .spe file (old and new format)."""
    try:
        h = BinFile(FileName, 'ieee-le')
    except OSError:
        raise ValueError(f'Could not open Magnettech spectrometer file {FileName}.')
    fileSize = os.path.getsize(FileName)
    nPts = [n for n in (512, 1024, 2048, 4096) if fileSize == 2 * n + 64]
    if not nPts:
        raise ValueError('This file does not have the correct file size for a Magnettech SPE file.')
    nPoints = nPts[0]
    Data = h.read(nPoints, 'int16').reshape(-1, 1)
    h.seek(fileSize - 4)
    FileFlags = int(h.scalar('uint8'))
    mwFreqAvailable = (FileFlags & 1) != 0
    oldSpeFormat = (FileFlags & 2) == 0
    temperatureAvailable = (FileFlags & 4) != 0
    div = 100 if oldSpeFormat else 1000

    def readfbs():
        x = h.read(2, 'int16')
        return x[0] + x[1] / div

    h.seek(2 * nPoints)
    P = Struct()
    P.B0_Field = readfbs() / 10
    P.B0_Scan = readfbs() / 10
    P.Modulation = readfbs() / 10000
    P.MW_Attenuation = readfbs()
    P.ScanTime = readfbs()
    GainMantissa = readfbs()
    GainExponent = readfbs()
    P.Gain = GainMantissa * 10.0 ** float(np.sign(GainExponent) * np.floor(abs(GainExponent) + 0.5))
    P.Number = readfbs()
    readfbs()
    P.Time_const = readfbs()
    readfbs()
    readfbs()
    P.NumberSamples = readfbs()
    if temperatureAvailable:
        P.Temperature = h.scalar('int32')
    else:
        h.skip(1, 'int32')
        P.Temperature = np.zeros((0, 0))
    readfbs()
    P.FileFlags = h.scalar('uint8')
    if mwFreqAvailable:
        mwf = h.read(3, 'uint8')
        P.mwFreq = (mwf[2] + 256 * mwf[1] + 256 ** 2 * mwf[0]) / 1e6
    else:
        P.mwFreq = np.zeros((0, 0))
    Abscissa = (P.B0_Field + np.linspace(-0.5, 0.5, Data.size) * P.B0_Scan).reshape(-1, 1)
    return Data, Abscissa, P
