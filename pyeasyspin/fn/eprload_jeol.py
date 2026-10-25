"""Reads in data from JEOL files"""
import numpy as np
from .._registry import nout
from .._struct import Struct
from .._io import BinFile

# (field, kind) lists; kind: int = C string of that length, 'i' int32, 'f' float32,
# ('skip', n) n chars skipped, ('skipi', n) n int32 skipped
_SPECTROMETER = (
    [('sType', 12), ('SpectrometerFlag', 'i')]
    + [(k, 16) for k in ('magnet water magPower magCurrent centerField sweepWidFin sweepWidCor sweepPulse '
                         'pulseNumber sweepFTime sweepBTime scanTime swCont sPosition ePosition pPosition '
                         'cPosition lfsBlock xPosition').split()]
    + [(None, ('skip', 112))]
    + [(k, 16) for k in ('modFreq modWidFin modWinCor phase Amp1_rcvMode Amp1_phasePlus Amp1_amplitudeFin '
                         'Amp1_amplitudeCor Amp1_timeConstant Amp1_zero Amp2_rcvMode Amp2_phasePlus '
                         'Amp2_amplitudeFin Amp2_amplitudeCor Amp2_timeConstant Amp2_zero amplock').split()]
    + [('sourceCH2', 8), (None, ('skip', 120)),
       ('uType', 8), ('uFreq', 16), ('uFreqUnit', 8), ('uPower', 16), ('uPowerUnit', 8), ('uPhase', 16),
       ('uCoupling', 16), ('slavePower', 16), ('slavePhase', 16), ('uRef', 16), ('autoTuneMode', 16),
       ('autoTune', 16), ('afc', 8), ('afcclk', 16), ('mod', 8), ('gunPower', 8), ('ref', 8),
       ('att_30dB', 8), ('afcbalance', 8), ('uDetM', 8), ('uBalM', 8), ('afcphase', 8), ('uStab', 8),
       ('shflock', 16), ('uFreqCal', 16), ('uFreqCalUnits', 8), ('RotationStepAngle', 10),
       ('RotationAngle', 10), ('RotationZero', 10), ('motorLow', 40), ('motorUp', 40), (None, ('skip', 18)),
       ('acqPoint', 8), ('dataLength', 8), ('acqPulsMode', 8), ('enType', 16)]
    + [(f'ch{c}_{k}', n) for c in (1, 2) for k, n in (('eLeftFreq', 16), ('eRightFreq', 16),
                                                        ('eCurrentFreq', 16), ('eFunit', 8), ('ePower', 16),
                                                        ('eDB', 12), ('ePunit', 8), ('eSweepTime', 12),
                                                        ('eModWidth', 8))]
    + [(None, ('skip', 8)), ('vtType', 16), ('temperature', 16), ('tempStart', 16), ('tempEnd', 16),
       ('tempStep', 16), ('tempUnit', 8), ('tempStatus', 8), ('tempError', 16), ('readyTime', 16),
       ('okTempRange', 16), ('tempControl', 8), ('tempLock', 16), ('endorLock', 16), ('selfCheck', 8),
       ('eCenterFreq', 16), ('eSweepFreq', 16), (None, ('skip', 48)), ('MarkerPos1', 16),
       ('MarkerPos2', 16), ('controlFA', 16), (None, ('skip', 112))])

_GEN_NORMAL = [('date', 32), ('accumulation', 'i'), ('preAccumulation', 'i'), ('delayTime', 'i'),
               ('intervalTime', 'i'), ('index', 'i'), ('repetition', 'i'), (None, ('skipi', 4)),
               ('sampleTime', 'i'), (None, ('skipi', 4)), ('accumWave', 8), ('baselineMode', 8),
               ('sampleMode', 8), ('sigTrig', 8), ('refFile', 64), ('paraFile', 64), ('chMode', 16),
               (None, ('skip', 112))]
_ENDOR_CH = [('eLeftFreq', 16), ('eRightFreq', 16), ('eCurrentFreq', 16), ('eFunit', 8), ('ePower', 16),
             ('eDB', 12), ('ePunit', 8), ('eSweepTime', 12), ('eModWidth', 16)]
_GEN_ENDOR = ([('date', 32), ('accumulation', 'i'), ('accumWave', 8), ('baselineMode', 8), ('chMode', 16),
               ('MagnetPosition', 16)]
              + [('ch1_' + k, n) for k, n in _ENDOR_CH] + [(None, ('skip', 248))]
              + [('ch2_' + k, n) for k, n in _ENDOR_CH] + [(None, ('skip', 248))]
              + [('eCenterFreq', 16), ('eSweepFreq', 16), (None, ('skip', 352))])
_AREA = [('xAreaFlag', 'i'), ('left', 'f'), ('right', 'f'), ('peak', 'f'), ('peakwidth', 'f'),
         ('peakMin', 'f'), ('peakMax', 'f'), ('pleft', 'i'), ('pright', 'i')]
_LINE = [('xLineFlag', 'i'), ('pos', 'f'), ('gvalue', 'f'), ('point', 'i')]


def _readcstring(h, n):
    ch = h.chars(n)
    i = ch.find('\0')
    return ch[:i] if i >= 0 else ''


def _fill(h, s, spec):
    for name, kind in spec:
        if isinstance(kind, tuple):
            h.skip(kind[1], 'char' if kind[0] == 'skip' else 'int32')
        elif kind == 'i':
            s[name] = h.scalar('int32')
        elif kind == 'f':
            s[name] = h.scalar('float32')
        else:
            s[name] = _readcstring(h, kind)
    return s


@nout(3)
def eprload_jeol(FileName=None):
    """(Data, Abscissa, Parameters) from a JEOL binary file. Parameters has the sub-structs
    Header, General, Spectrometer, Generator, ProcessPara, Calculator, Data (as EasySpin)."""
    try:
        h = BinFile(FileName, 'ieee-le')
    except OSError:
        raise ValueError(f'Could not open {FileName}.')
    hd = Struct()
    hd.processType = _readcstring(h, 16)
    hd.filename = _readcstring(h, 64)
    h.skip(1, 'int32')
    hd.auxNum = h.scalar('int32')
    hd.dataSet = h.scalar('int32')
    hd.dataLength = h.scalar('int32')
    hd.dataKind = _readcstring(h, 15)
    hd.dispPart = h.chars(1)
    hd.xOffset = h.scalar('float32')
    hd.xRange = h.scalar('float32')
    hd.xUnit = _readcstring(h, 16)
    hd.yMin = h.scalar('float32')
    hd.yMax = h.scalar('float32')
    hd.yUnit = _readcstring(h, 16)
    hd.zPoint = h.scalar('float32')
    hd.zUnit = _readcstring(h, 16)
    for k in ('xStart', 'xStop', 'yStart', 'yStop', 'zStart', 'zStop'):
        hd[k] = h.scalar('float32')
    h.skip(88)
    hd.waveIndex = h.scalar('int32')
    h.skip(28)
    jeol_endor = hd.processType == 'endor'
    h.skip(4096, 'int8')
    h.skip(16)

    gen = Struct()
    for k, n in (('system', 32), ('version', 16), ('date', 32), ('title', 128), ('sample', 128),
                 ('comment', 128)):
        gen[k] = _readcstring(h, n)
    h.skip(224)
    gen.previousINFO_filename = _readcstring(h, 64)
    gen.previousINFO_dir = _readcstring(h, 512)
    h.skip(1, 'int32')
    gen.previousINFO_processType = _readcstring(h, 16)
    gen.originalINFO_filename = _readcstring(h, 64)
    gen.originalINFO_dir = _readcstring(h, 512)
    h.skip(1, 'int32')
    gen.originalINFO_processType = _readcstring(h, 16)

    spec = _fill(h, Struct(), _SPECTROMETER)
    genr = _fill(h, Struct(), _GEN_ENDOR if jeol_endor else _GEN_NORMAL)

    pp = Struct()
    for k in ('res_q', 'micro_hz', 'efect_pls', 'nike_hz', 'pulser', 'digitzer', 'resonator'):
        pp[k] = h.scalar('float32')
    pp.fftCount = h.scalar('int32')
    pp.lastProcessTyp = _readcstring(h, 16)
    for pre in ('std_', ''):
        pp[pre + 'spinFlag'] = h.scalar('int32')
        for k in ('marleft', 'marright', 'marintegval', 'raw_marintegval', 'sigleft', 'sigright',
                  'sigintegval', 'raw_sigintegval', 'spinnum'):
            pp[pre + k] = h.scalar('float32')
        h.skip(4, 'float32')
    pp.xAreaM = _fill(h, Struct(), _AREA)
    pp.xAreaS = _fill(h, Struct(), _AREA)
    pp.xArea = [_fill(h, Struct(), _AREA) for _ in range(8)]
    pp.xLinem1 = _fill(h, Struct(), _LINE)
    pp.xLinem2 = _fill(h, Struct(), _LINE)
    pp.xLine = [_fill(h, Struct(), _LINE) for _ in range(8)]
    pp.aope = list(h.chars(4))   # 4x1 char column (a list of row strings)
    pp.modFilename = _readcstring(h, 16)
    pp.yLine = h.scalar('int32')
    pp.calibFlag = h.scalar('int32')
    h.skip(128)

    calc = Struct()
    pt = hd.processType
    if pt == 'yZero':
        h.skip(160)
    elif pt == 'xZero':
        calc.xzero = h.scalar('float32')
    elif pt == 'yGain':
        calc.gain = h.scalar('float32')
    elif pt == 'xShift':
        calc.shiftvalue = h.scalar('float32')
    elif pt == 'reverse':
        calc.reversevalue = h.scalar('float32')
    elif pt == 'fill':
        calc.fillingValue = h.scalar('float32')
        calc.oriEnd = h.scalar('int32')
        calc.length = h.scalar('int32')
    elif pt == 'window':
        calc.parcent = h.read(4, 'float32').reshape(-1, 1)
        calc.xPoint = h.read(4, 'float32').reshape(-1, 1)
        calc.xValue = h.read(4, 'float32').reshape(-1, 1)
        calc.function = _readcstring(h, 32)
    elif pt in ('diff', 'inte'):
        calc.inteFlag = h.scalar('int32')
        calc.order = h.scalar('int32')
        for k in ('points', 'pLeft', 'pRight', 'xLeft', 'xRight', 'baseline'):
            calc[k] = h.scalar('float32')
    elif pt == 'smooth':
        calc.ummy = _readcstring(h, 16)
    elif pt == 'fit':
        h.skip(90120)
    elif pt in ('fRtime', 'tRtime'):
        h.skip(4264)
    elif pt == 'phase':
        calc.theta = h.scalar('float32')
        calc.radian = h.scalar('float32')
        calc.comment = _readcstring(h, 256)
        h.skip(544)
    elif pt == 'mT2MHz':
        h.skip(4)
    elif pt == 'hw':
        h.skip(40)
    elif pt == 'mem':
        h.skip(2708)

    auxNum, dataSet, dataKind = hd.auxNum, hd.dataSet, hd.dataKind
    N = int(hd.dataLength)
    jd = Struct()
    Abscissa = np.zeros((0, 0))
    Data = np.zeros((0, 0))
    absc = lambda: (hd.xOffset + np.linspace(0, hd.xRange, N)).reshape(1, -1)
    if dataSet == 1:
        jd.CH1_DATA = h.read(N, 'float32').reshape(-1, 1)
        if dataKind == 'complex':
            jd.CH2_DATA = h.read(N, 'float32').reshape(-1, 1)
        Abscissa = absc()
    elif auxNum in (1, 12) and dataKind == 'complex':
        nv, ni = (1, 8) if auxNum == 1 else (12, 96)
        nset = int(dataSet)
        sv, si, c1, c2 = [], [], [], []
        for _ in range(nset):
            sv.append(h.read(nv, 'float32'))
            si.append(h.chars(ni))
            c1.append(h.read(N, 'float32'))
            c2.append(h.read(N, 'float32'))
        jd.SEQ_VALUE = np.array(sv)
        jd.SEQ_ITEM = si if nset > 1 else si[0]
        jd.CH1_DATA = np.array(c1)
        jd.CH2_DATA = np.array(c2)
        Abscissa = absc()
    if dataKind == 'real':
        Data = jd.CH1_DATA
    elif dataKind == 'complex':
        Data = jd.CH1_DATA + 1j * jd.CH2_DATA

    Parameters = Struct(Header=hd, General=gen, Spectrometer=spec, Generator=genr, ProcessPara=pp,
                        Calculator=calc, Data=jd)
    return Data, Abscissa, Parameters
