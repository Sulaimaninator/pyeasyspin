"""Element number to element symbol."""
_SYMBOLS = 'H HeLiBeB C N O F NeNaMgAlSiP S ClArK CaScTiV CrMnFeCoNiCuZnGaGeAsSeBrKrRbSrY ZrNbMoTcRuRhPdAgCdInSnSbTeI XeCsBaLaCePrNdPmSmEuGdTbDyHoerTmYbLuHfTaW ReOsIrPtAuHgTlPbBiPoAtRnFrRaAcThPaU NpPuAmCmBkCfEsFmMdNoLrRfDbSgBhHsMtDsRgCn'


def elementno2symbol(no=None):
    """Symbol of element number no (1..112); '' if out of range."""
    if no < 1 or no > 112:
        return ''
    i = int(no) * 2 - 2
    s = _SYMBOLS[i:i + 2]
    return s[0] if s[1] == ' ' else s
