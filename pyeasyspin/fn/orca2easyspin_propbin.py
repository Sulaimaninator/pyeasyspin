"""Read ORCA 3.x binary .prop file"""
import warnings
import numpy as np
from .._registry import F
from .._struct import Struct
from .._io import BinFile
from .orca2easyspin_maintxt import setrows

_PROPNAMES = {0: 'SCF_DIP', 1: 'SCF_POL', 2: 'SCF_ABS', 3: 'SCF_CD', 4: 'SCF_D', 5: 'SCF_G', 6: 'SCF_A',
              7: 'SCF_Q', 8: 'SCF_SHIFT', 9: 'SCF_RHO_0', 10: 'MRCI_DIP', 11: 'MRCI_ABS', 12: 'MRCI_CD',
              13: 'MRCI_D', 14: 'MRCI_G', 15: 'MRCI_A', 16: 'MRCI_Q', 17: 'XYZ', 18: 'ATNO',
              19: 'CASSCF_DIP', 20: 'CASSCF_ABS', 21: 'CASSCF_CD', 22: 'CASSCF_D', 23: 'CASSCF_G',
              24: 'CASSCF_A', 25: 'CASSCF_Q', 26: 'NEVPT2_DIP', 27: 'NEVPT2_ABS', 28: 'NEVPT2_CD',
              29: 'NEVPT2_D', 30: 'NEVPT2_G', 31: 'NEVPT2_A', 32: 'NEVPT2_Q', 33: 'CASSCFCUSTOME_DIP',
              34: 'CASSCFCUSTOME_ABS', 35: 'CASSCFCUSTOME_CD', 36: 'CASSCFCUSTOME_D',
              37: 'CASSCFCUSTOME_G', 38: 'CASSCFCUSTOME_A', 39: 'CASSCFCUSTOME_Q', 40: 'QDNEVPT2_DIP',
              41: 'QDNEVPT2_ABS', 42: 'QDNEVPT2_CD', 43: 'QDNEVPT2_D', 44: 'QDNEVPT2_G', 45: 'QDNEVPT2_A',
              46: 'QDNEVPT2_Q', -1: 'termination'}

_QREFMASSNO = [
    2, 0, 7, 9, 11, 0, 14, 17, 0, 21,
    23, 25, 27, 0, 0, 33, 35, 0, 39, 43,
    45, 47, 51, 53, 55, 0, 59, 61, 63, 67,
    69, 73, 75, 0, 79, 83, 85, 87, 0, 91,
    93, 95, 0, 101, 0, 105, 0, 0, 115, 0,
    121, 0, 127, 131, 133, 137, 139, 0, 141, 143,
    0, 147, 153, 157, 159, 163, 165, 167, 0, 173,
    175, 177, 181, 0, 187, 189, 193, 0, 197, 201,
    0, 0, 209, 0, 0, 0, 0, 0, 227, 0,
    0, 235, 237, 0, 243, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0]


def _c(x):
    return x[0] if isinstance(x, tuple) else x


def propertystring(ID):
    return _PROPNAMES.get(int(ID), 'unknown')


def efg2Q(efgpv, elem, units):
    """Quadrupole principal values (MHz) from EFG principal values of element elem."""
    efgpv = np.asarray(efgpv, dtype=float)
    if units == 'SI':
        pass
    elif units == 'au':
        efgpv = efgpv * (_c(F.hartree()) / F.evolt() / _c(F.bohrrad()) ** 2)
    else:
        raise ValueError("Unrecognized unit '%s' for EFG. Use 'SI' or 'au'." % units)
    massNo = _QREFMASSNO[int(elem) - 1]
    if massNo == 0:
        return np.zeros(3)
    qrefIso = '%d%s' % (massNo, F.elementno2symbol(elem))
    eQ = F.echarge() * F.nucqmom(qrefIso) * F.barn()
    I = F.nucspin(qrefIso)
    Qpv_ = eQ / 2 / I / (2 * I - 1) * efgpv
    return Qpv_ / F.planck() / 1e6


def orca2easyspin_propbin(propfilename=None):
    """Sys from a binary ORCA .prop file (S, xyz, g/gFrame, D/DFrame, Nucs as a string,
    NucsIdx, A/AFrame, Q/QFrame, data). Returns [] if the file holds only the terminator."""
    PropertyIDtype, rowcoltype, datatype = 'int32', 'int32', 'float64'
    terminateID = -1
    maxPropertyID = 53
    try:
        f = BinFile(propfilename, 'ieee-le')
    except OSError as e:
        raise ValueError('Could not open file %s: %s' % (propfilename, e.strerror)) from None
    first = f.scalar(PropertyIDtype)
    if not isinstance(first, float):
        raise ValueError('Operands to the logical AND (&&) and OR (||) operators must be convertible '
                         'to logical scalar values.')
    if first == terminateID:
        return np.zeros((0, 0))
    bo = 'ieee-be' if first > maxPropertyID else 'ieee-le'
    f = BinFile(propfilename, bo)

    S = None
    xyz = Charge = Dpv = DFrame = gpv = gFrame = None
    Apv = AFrame = efg = efgFrame = rho0 = None
    Atoms = np.zeros((0, 0))
    while True:      # while ~feof(f): feof is only set by a read past the end
        PropertyID = f.scalar(PropertyIDtype)
        if not isinstance(PropertyID, float):
            # fread at the end of the file returns []; EasySpin's || then errors
            raise ValueError('Operands to the logical AND (&&) and OR (||) operators must be '
                             'convertible to logical scalar values.')
        if PropertyID == terminateID:
            break
        if PropertyID < 0 or PropertyID > maxPropertyID:
            raise ValueError('Unknown property encountered (PropertyID = %d)' % PropertyID)
        nRows = int(f.scalar(rowcoltype))
        nColumns = int(f.scalar(rowcoltype))
        nElements = nRows * nColumns
        raw = f.read(nElements, datatype)
        data = raw.reshape(nColumns, nRows, order='F')
        dl = raw                                     # data(:) for linear indexing
        pid = int(PropertyID)
        if pid == 40:
            Charge = dl[0]
        elif pid == 41:
            S = (dl[0] - 1) / 2
        elif pid in (17, 42):
            xyz = dl.reshape(3, -1, order='F').T
            xyz = xyz * _c(F.bohrrad()) / 1e-10
        elif pid in (18, 43):
            Atoms = data.T
        elif pid in (5, 14, 23, 30, 27):
            gpv = dl[0:3].reshape(1, 3)
            if np.all(gpv == 0):
                warnings.warn('Property entry with ID %d (%s) is all zero.' % (pid, propertystring(pid)))
                gFrame = np.zeros(3)
            else:
                R = dl[3:12].reshape(3, 3, order='F')
                gFrame = F.eulang(R.T)
        elif pid in (4, 13, 22, 29, 36):
            Dpv = dl[2:5].reshape(1, 3)
            Dpv = Dpv * 1e-4 * F.clight()
            R = dl[5:14].reshape(3, 3, order='F')
            DFrame = F.eulang(R.T)
        elif pid in (6, 15, 24, 31, 38):
            nucIdx = data[0, :] + 1
            for iNuc in range(nucIdx.size):
                idx = int(nucIdx[iNuc]) - 1
                Apv = setrows(Apv, idx, data[2:5, iNuc])
                R = data[5:14, iNuc].reshape(3, 3, order='F').T
                AFrame = setrows(AFrame, idx, F.eulang(R.T))
        elif pid in (7, 16, 25, 32, 39):
            nucIdx = data[0, :] + 1
            for iNuc in range(nucIdx.size - 1, -1, -1):
                idx = int(nucIdx[iNuc]) - 1
                efg_au = data[1:4, iNuc]
                efg = setrows(efg, idx, efg_au * (_c(F.hartree()) / F.echarge() / _c(F.bohrrad()) ** 2))
                R = data[4:13, iNuc].reshape(3, 3, order='F').T
                efgFrame = setrows(efgFrame, idx, F.eulang(R.T))
        elif pid == 9:
            nucIdx = (data[0, :] + 1).astype(int)
            rho0_ = data[1, :]
            n = int(nucIdx.max()) if nucIdx.size else 0
            r = np.zeros(max(n, 0 if rho0 is None else rho0.size))
            if rho0 is not None:
                r[:rho0.size] = rho0
            r[nucIdx - 1] = rho0_ * (_c(F.bohrrad()) / 1e-10) ** 3
            rho0 = r

    nAtoms = int(np.size(Atoms))
    info = Struct()
    Sys = Struct()
    if S is not None:
        Sys.S = S
    if xyz is not None and xyz.size:
        Sys.xyz = xyz
        info.NucId = Atoms
    if Charge is not None:
        info.Charge = Charge
    if gpv is not None:
        Sys.g = gpv
        Sys.gFrame = gFrame
    if Dpv is not None:
        Sys.D = Dpv
        Sys.DFrame = DFrame
    anyHyperfine = Apv is not None and Apv.size > 0
    anyQuadrupole = efg is not None and efg.size > 0
    if anyHyperfine:
        if Apv.shape[0] < nAtoms:
            Apv = setrows(Apv, nAtoms - 1, np.zeros(Apv.shape[1]))
        if AFrame.shape[0] < nAtoms:
            AFrame = setrows(AFrame, nAtoms - 1, np.zeros(AFrame.shape[1]))
    if anyQuadrupole:
        if efg.shape[0] < nAtoms:
            efg = setrows(efg, nAtoms - 1, np.zeros(efg.shape[1]))
        if efgFrame.shape[0] < nAtoms:
            efgFrame = setrows(efgFrame, nAtoms - 1, np.zeros(efgFrame.shape[1]))
        Qpv = np.zeros(efg.shape)
    if nAtoms > 0:
        At = Atoms.ravel(order='F')
        if anyQuadrupole:
            for iAtom in range(nAtoms - 1, -1, -1):
                if not np.any(efg[iAtom, :]):
                    continue
                Qpv[iAtom, :] = efg2Q(efg[iAtom, :], At[iAtom], 'SI')
        NucStr = ''
        for iAtom in range(nAtoms):
            NucStr = NucStr + ',' + F.elementno2symbol(At[iAtom])
        if NucStr:
            NucStr = NucStr[1:]
        Sys.Nucs = NucStr
        Sys.NucsIdx = np.arange(1, nAtoms + 1, dtype=float).reshape(1, -1)
        if anyHyperfine:
            Sys.A = Apv
            Sys.AFrame = AFrame
        if anyQuadrupole:
            Sys.Q = Qpv
            Sys.QFrame = efgFrame
    Sys.data = info
    return Sys
