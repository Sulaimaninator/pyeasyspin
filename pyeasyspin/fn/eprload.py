"""Load experimental EPR data"""
import os
import re
from .._registry import F, nout


def _fileparts(f):
    """MATLAB fileparts on Windows ('/' and '\\' are separators)."""
    i = max(f.rfind('/'), f.rfind('\\'))
    p, name = (f[:i], f[i + 1:]) if i >= 0 else ('', f)
    if i == 0:
        p = f[0]
    j = name.rfind('.')
    if j >= 0:
        return p, name[:j], name[j:]
    return p, name, ''


def _fullfile(p, name):
    """MATLAB fullfile(p, name): joined with the platform separator, '/' normalised to it."""
    if not p:
        return name
    sep = os.sep
    s = p if p.endswith(('/', '\\')) else p + sep
    s = s + name
    if sep == '\\':
        s = s.replace('/', '\\')
    return s


_FORMATS = {'.DTA': 'BrukerBES3T', '.DSC': 'BrukerBES3T', '.PAR': 'BrukerESP', '.SPC': 'BrukerESP',
            '.D01': 'SpecMan', '.SPE': 'MagnettechBinary', '.XML': 'MagnettechXML',
            '.ESR': 'ActiveSpectrum', '.DAT': 'AdaniDAT', '.JSON': 'AdaniJSON', '.ECO': 'qese/tryscore',
            '.PLT': 'MAGRES', '.SPK': 'VarianETH', '.REF': 'VarianETH', '.D00': 'WeizmannETH'}


@nout(-1)
def eprload(FileName=None, Scaling=None, nargout=None):
    """eprload(FileName, Scaling='', nargout=3).

    nargout=1 -> Data; 2 -> (Abscissa, Data); 3 (default) -> (Abscissa, Data, Parameters);
    4 -> (..., FileName). Data/Abscissa are lists for several data sets/axes (MATLAB cells).
    No file-selection dialog and no plotting (MATLAB does those for a folder / nargout=0)."""
    if nargout is None or nargout == 0:
        nargout = 3
    if nargout < 0 or nargout > 4:
        raise ValueError('Please provide 1, 2, 3 or 4 output arguments!')
    if FileName is None:
        FileName = os.getcwd()
    if Scaling is None:
        Scaling = ''
    FileName = str(FileName)
    if os.path.isdir(FileName):
        raise ValueError('eprload: FileName is a folder; the file-selection dialog is not available '
                         'in Python. Give a file name.')

    p, Name, FileExtension = _fileparts(FileName)
    FullBaseName = _fullfile(p, Name)
    if not FileExtension or len(FileExtension) < 2:
        if os.path.exists(FullBaseName + '.dta'):
            FileExtension = '.dta'
        if os.path.exists(FullBaseName + '.DTA'):
            FileExtension = '.DTA'
        if os.path.exists(FullBaseName + '.spc'):
            FileExtension = '.spc'
    FileName = FullBaseName + FileExtension
    if not os.path.exists(FileName):
        raise ValueError(f'The file or directory {FileName} does not exist!')

    FileFormat = _FORMATS.get(FileExtension.strip().upper())
    if FileFormat is None:
        try:
            with open(FileName, 'rb') as h:
                processType = h.read(16).decode('latin-1')
        except OSError:
            raise ValueError('Could not open ' + FileName)
        if re.search('^spin|^cAcqu|^endor|^pAcqu|^cidep|^sod|^iso|^ani', processType):
            FileFormat = 'JEOL'
        else:
            raise ValueError(f'Unsupported file extension {FileExtension}')

    if Scaling and FileFormat in ('BrukerBES3T', 'BrukerESP'):
        if any(c not in 'nPGTc' for c in Scaling):
            raise ValueError("Scaling can only contain 'n', 'P', 'G', 'T', and 'c'.")
    elif Scaling:
        raise ValueError('Scaling does not work for this file type.')

    if FileFormat == 'BrukerBES3T':
        Data, Abscissa, Parameters = F.eprload_BrukerBES3T(FullBaseName, FileExtension, Scaling)
    elif FileFormat == 'BrukerESP':
        Data, Abscissa, Parameters = F.eprload_BrukerESP(FullBaseName, FileExtension, Scaling)
    else:
        loader = {'SpecMan': 'eprload_specman', 'MagnettechBinary': 'eprload_MagnettechBinary',
                  'MagnettechXML': 'eprload_MagnettechXML', 'ActiveSpectrum': 'eprload_ActiveSpectrum',
                  'AdaniDAT': 'eprload_AdaniDAT', 'AdaniJSON': 'eprload_AdaniJSON', 'JEOL': 'eprload_jeol',
                  'qese/tryscore': 'eprload_qeseETH', 'MAGRES': 'eprload_MAGRES',
                  'VarianETH': 'eprload_VarianE9ETH', 'WeizmannETH': 'eprload_d00WISETH'}[FileFormat]
        Data, Abscissa, Parameters = getattr(F, loader)(FileName)

    if nargout == 1:
        return Data
    if nargout == 2:
        return Abscissa, Data
    if nargout == 3:
        return Abscissa, Data, Parameters
    return Abscissa, Data, Parameters, FileName
