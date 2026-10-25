"""Add default fields to a user structure."""
from .._struct import Struct
from .._mat import isempty


def adddefaults(UserStruct=None, DefaultStruct=None):
    """Return DefaultStruct with every field of UserStruct copied over it. A field that differs
    from a default field name only in case is an error."""
    Supplemented = Struct(DefaultStruct or {})
    if isempty(UserStruct):
        return Supplemented
    UserStruct = _spellcheckfields(UserStruct, list((DefaultStruct or {}).keys()))
    Supplemented = Struct(DefaultStruct or {})
    if not isempty(UserStruct):
        for name, value in UserStruct.items():
            Supplemented[name] = value
    return Supplemented


def _spellcheckfields(Structure, ExactFieldNames):
    if isempty(Structure):
        return None
    out = Struct()
    lower = [n.lower() for n in ExactFieldNames]
    for name in Structure:
        correct = name in ExactFieldNames
        anycase = name.lower() in lower
        if not correct and anycase:
            right = ExactFieldNames[lower.index(name.lower())]
            raise ValueError(f"\n   Supplied field '{name}' is not correct. Did you mean '{right}'?\n")
        out[name] = Structure[name]
    return out
