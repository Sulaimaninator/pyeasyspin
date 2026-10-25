"""Element symbol to element number."""
import numpy as np
from .elementno2symbol import _SYMBOLS


def elementsymbol2no(symbol=None):
    """Element number of a symbol (as EasySpin: substring search in the symbol table)."""
    symbol = str(symbol)
    if len(symbol) == 1:
        symbol = symbol + ' '
    elif len(symbol) > 2:
        symbol = symbol[:2]
    idx = [i + 1 for i in range(len(_SYMBOLS) - 1) if _SYMBOLS[i:i + 2] == symbol]
    no = [(i + 1) / 2 for i in idx]
    if len(no) == 1:
        return no[0]
    return np.array(no, dtype=float).reshape(1, -1) if no else np.zeros((1, 0))
