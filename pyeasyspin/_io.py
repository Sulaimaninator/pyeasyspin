"""Helpers shared by the file I/O functions: MATLAB text-scanning semantics (sscanf %f,
strtok, deblank, genvarname) and a small binary reader with fread semantics."""
import keyword as _kw
import os
import re
import numpy as np

_WS = ' \t\n\r\v\f'
_FLOAT = re.compile(r'[+-]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)', re.I)
_MATLAB_KEYWORDS = {'break', 'case', 'catch', 'classdef', 'continue', 'else', 'elseif', 'end', 'for',
                    'function', 'global', 'if', 'otherwise', 'parfor', 'persistent', 'return', 'spmd',
                    'switch', 'try', 'while'}


def sscanf_f(s, count=None):
    """MATLAB sscanf(s, '%f') (also %e, %g): (values list, errmsg). Stops at the first
    unparsable token; errmsg is non-empty then (as MATLAB's third output)."""
    vals, pos, n = [], 0, len(s)
    while pos < n and (count is None or len(vals) < count):
        while pos < n and s[pos] in _WS:
            pos += 1
        if pos >= n:
            break
        m = _FLOAT.match(s, pos)
        if not m:
            return vals, 'Matching failure in format.'
        vals.append(float(m.group(0)))
        pos = m.end()
    return vals, ''


def num(s):
    """sscanf(s,'%f') as MATLAB uses it for a scalar keyword: a float if one value, else array
    (empty array if none)."""
    v, _ = sscanf_f(s)
    if len(v) == 1:
        return v[0]
    return np.array(v, dtype=float).reshape(-1, 1) if v else np.zeros((0, 1))


def strtok(s, delims=None):
    """MATLAB strtok: skip leading delimiters, token up to the next delimiter, remainder from it."""
    if delims is None:
        delims = _WS + '\0' + ''.join(chr(c) for c in range(1, 33) if chr(c) not in _WS)
    i, n = 0, len(s)
    while i < n and s[i] in delims:
        i += 1
    j = i
    while j < n and s[j] not in delims:
        j += 1
    return s[i:j], s[j:]


def deblank(s):
    """MATLAB deblank: remove trailing whitespace and null characters."""
    return s.rstrip(_WS + '\0')


def isletter(c):
    return c.isalpha()


def isvarname(k):
    return (bool(re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', k)) and len(k) <= 63
            and k not in _MATLAB_KEYWORDS)


def genvarname(k):
    """MATLAB genvarname for one name (no exclusion list): invalid characters become their hex
    code '0xHH', a leading non-letter gets an 'x' prefix, keywords become 'x' + Capitalized."""
    if k in _MATLAB_KEYWORDS:
        return 'x' + k[0].upper() + k[1:]
    out = ''.join(c if re.fullmatch(r'[A-Za-z0-9_]', c) else '0x%02X' % ord(c) for c in k)
    if not out or not out[0].isalpha():
        out = 'x' + out
    return out[:63]


def read_text(path):
    """File contents as str (UTF-8, falling back to Latin-1 byte-per-char)."""
    with open(path, 'rb') as fh:
        b = fh.read()
    try:
        return b.decode('utf-8')
    except UnicodeDecodeError:
        return b.decode('latin-1')


def textscan_lines(path):
    """textscan(fh,'%s','whitespace','','delimiter','\\n'){1}: lines without EOL characters."""
    t = read_text(path).replace('\r\n', '\n').replace('\r', '\n')
    lines = t.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    return lines


def fgetl_lines(path):
    """Lines as successive fgetl calls return them (trailing '\\r' kept), until feof."""
    t = read_text(path)
    lines = t.split('\n')
    if lines and lines[-1] == '':
        lines.pop()
    return lines


def file_exists(path):
    """exist(path,'file') > 0 for files and folders."""
    return os.path.exists(path)


_PREC = {'int8': 'i1', 'uint8': 'u1', 'int16': 'i2', 'uint16': 'u2', 'int32': 'i4', 'uint32': 'u4',
         'int64': 'i8', 'float32': 'f4', 'float': 'f4', 'single': 'f4', 'float64': 'f8',
         'double': 'f8', 'char': 'u1'}


class BinFile:
    """Binary file with MATLAB fread semantics (numbers come back as float64 columns)."""

    def __init__(self, path, byteorder='ieee-le'):
        with open(path, 'rb') as fh:
            self.b = fh.read()
        self.pos = 0
        self.bo = '>' if str(byteorder).lower().startswith('ieee-be') or byteorder in ('b',) else '<'

    def read(self, n, prec):
        """fread(h, n, prec) as a float64 1-D array (n=None/inf reads to the end)."""
        dt = np.dtype(self.bo + _PREC[prec])
        avail = (len(self.b) - self.pos) // dt.itemsize
        cnt = avail if n is None or n == np.inf else min(int(n), avail)
        a = np.frombuffer(self.b, dtype=dt, count=cnt, offset=self.pos)
        self.pos += cnt * dt.itemsize
        return a.astype(np.float64)

    def scalar(self, prec):
        """fread(h, 1, prec): float, or empty (0,1) array at end of file."""
        a = self.read(1, prec)
        return float(a[0]) if a.size else np.zeros((0, 1))

    def chars(self, n):
        """fread(h, n, '*char').' as str (one byte per character)."""
        cnt = min(int(n), len(self.b) - self.pos)
        s = self.b[self.pos:self.pos + cnt].decode('latin-1')
        self.pos += cnt
        return s

    def skip(self, n, prec='char'):
        self.read(n, prec)

    def seek(self, offset):
        self.pos = int(offset)
