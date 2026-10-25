"""Convert an isotope string ('1H,14N') to a list of isotope names."""
import numpy as np


def nucstring2list(NucString=None, ParseMode=None):
    """List of isotope strings. ParseMode 's' (default) splits at every comma, 'm' only at
    commas outside parentheses (isotope mixtures)."""
    if isinstance(NucString, (list, tuple)):
        return NucString
    if NucString is None or (not isinstance(NucString, str) and np.size(NucString) == 0) or NucString == '':
        return []
    if ParseMode is None:
        ParseMode = 's'
    if ParseMode == 's':
        single = True
    elif ParseMode == 'm':
        single = False
    else:
        raise ValueError("Unknown parse mode - should be 's' or 'm'.")
    s = NucString
    if not isinstance(s, str):
        raise ValueError('Isotope list must be a string, not numeric.')
    if ';' in s:
        raise ValueError(f"Use , and not ; in nuclear spin string. You gave '{s}'.")
    s = ''.join(ch for ch in s if not ch.isspace())
    if s.endswith(','):
        s = s[:-1]
    if single:
        commas = [i for i, ch in enumerate(s) if ch == ',']
    else:
        lev, commas = 0, []
        for i, ch in enumerate(s):
            lev += (ch == '(') - (ch == ')')
            if ch == ',' and lev == 0:
                commas.append(i)
    bounds = [-1] + commas + [len(s)]
    Nucs = [s[bounds[k] + 1:bounds[k + 1]] for k in range(len(bounds) - 1)]
    return [n for n in Nucs if n]
