"""Loop over components and isotopologues, summing (or stacking) their spectra."""
import numpy as np

from .._registry import F, nout
from .._struct import Struct


def _name(f):
    return getattr(f, '__name__', '')


@nout(3)
def compisoloop(simfcn=None, Sys=None, Exp=None, Opt=None, autoRange=None, thirdOutput=None,
                separateSpectra=None):
    """(x, spec, info): simfcn (pepper, garlic, salt, chili, curry, saffron, ...) is called with
    nargout=3 for each isotopologue of each component (Sys_.singleiso = True), and the spectra are
    weighted by the isotopologue abundance."""
    includeInverseDomain = _name(simfcn) == 'saffron'
    info = Struct()
    if not isinstance(Sys, (list, tuple)):
        Sys = [Sys]
    nComponents = len(Sys)
    F.logmsg(1, '  number of component spin systems: %d', nComponents)
    SysList, nIsotopologues = [], []
    for c in range(nComponents):
        iso = F.isotopologues(Sys[c], Opt.IsoCutoff)
        iso = iso if isinstance(iso, list) else [iso]
        SysList.append(iso)
        nIsotopologues.append(len(iso))
        F.logmsg(1, '    component %d: %d isotopologues', c + 1, len(iso))
    nIsotopologues = np.array(nIsotopologues)
    if bool(autoRange) and int(np.sum(nIsotopologues)) > 1:
        if 'FrequencySweep' in Exp:
            str_ = 'Exp.mwRange or Exp.mwCenterSweep' if bool(Exp.FrequencySweep) else 'Exp.Range or Exp.CenterSweep'
        elif _name(simfcn) == 'salt':
            str_ = 'Exp.Range'
        else:
            str_ = ''
        raise ValueError('For multiple components, EasySpin cannot automatically determine a sweep range.\n'
                         f' Please specify sweep range manually using {str_}.')
    separateSpectra = bool(separateSpectra)
    thirdOutput = bool(thirdOutput)
    spec_list, fd_list = [], []
    spec = 0
    data_invdomain = 0
    anyMulti = bool(np.any(nIsotopologues > 1))
    if thirdOutput:
        info = Struct(Component=np.zeros((1, 0)), Isotopologue=np.zeros((1, 0)))
        if anyMulti:
            info.Nucs = []
        info.Transitions = []
        idx = 0
    info_, fdProvided = Struct(), False
    for iComponent in range(nComponents):
        for iIsotopologue in range(nIsotopologues[iComponent]):
            Sys_ = Struct(SysList[iComponent][iIsotopologue])
            Sys_.singleiso = True
            x, spec_, info_ = simfcn(Sys_, Exp, Opt, nargout=3)[:3]
            info_ = info_ if isinstance(info_, dict) else Struct()
            fdProvided = 'fd' in info_
            w = float(np.asarray(Sys_.weight).ravel()[0])
            if separateSpectra:
                spec_list.append(np.atleast_2d(np.asarray(spec_) * w))
                if includeInverseDomain and fdProvided:
                    fd_list.append(np.atleast_2d(np.asarray(info_.fd) * w))
            else:
                spec = spec + np.asarray(spec_) * w
                if includeInverseDomain and fdProvided:
                    data_invdomain = data_invdomain + np.asarray(info_.fd) * w
            if thirdOutput:
                idx += 1
                info.Component = np.append(info.Component, iComponent + 1).reshape(1, -1)
                info.Isotopologue = np.append(info.Isotopologue, iIsotopologue + 1).reshape(1, -1)
                if anyMulti:
                    info.Nucs.append(Sys_.Nucs if 'Nucs' in Sys_ else '')
                if 'Transitions' in info_:
                    info.Transitions.append(info_.Transitions)
                    nt = np.atleast_2d(np.asarray(info_.Transitions)).shape[0]
                    info.nTransitions = np.append(info.get('nTransitions', np.zeros((1, 0))), nt).reshape(1, -1)
                for fld in ('nSites', 'nOrientations'):
                    if fld in info_:
                        info[fld] = np.append(info.get(fld, np.zeros((1, 0))), info_[fld]).reshape(1, -1)
                if 'resfields' in info_:
                    info.setdefault('resfields', []).append(info_.resfields)
    if separateSpectra:
        spec = np.vstack(spec_list) if spec_list else np.zeros((0, 0))
        if includeInverseDomain:
            data_invdomain = np.vstack(fd_list) if fd_list else np.zeros((0, 0))
    if thirdOutput:
        if len(info.Transitions) == 1:
            info.Transitions = info.Transitions[0]
        if isinstance(info.Transitions, list) and len(info.Transitions) == 0:
            del info['Transitions']
        if 'resfields' in info and len(info.resfields) == 1:
            info.resfields = info.resfields[0]
        if includeInverseDomain:
            if 'SinglePointDetection' in info_:
                info.SinglePointDetection = info_.SinglePointDetection
            else:
                info_ = Struct(info_)
                if 'td' in info_ and np.size(info_.td) > 0:
                    info_.td = spec
                if fdProvided:
                    info_.fd = data_invdomain
                for k in info_:
                    info[k] = info_[k]
    return x, spec, info
