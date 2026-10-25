"""Print a log message if its level is at or below config.LogLevel."""
from ..config import config, log


def logmsg(*varargin):
    """logmsg(level, fmt, *args): MATLAB fprintf-style format, printed via the pyeasyspin logger."""
    if config.LogLevel is None:
        return
    level = varargin[0]
    # EasySpin calls logmsg(-Inf, 'WARNING ...') for messages that must always show, but its own
    # check (level < 0 -> error) turns them into crashes. -Inf is accepted here (DEVIATIONS.md).
    always = isinstance(level, (int, float)) and level == float('-inf')
    if not always and (not isinstance(level, (int, float)) or level % 1 != 0 or level < 0):
        raise ValueError('The first input to logmsg must be a non-negative integer.')
    if not always and level > config.LogLevel:
        return
    if len(varargin) < 2:
        raise ValueError('At least two input arguments expected.')
    fmt, args = varargin[1], varargin[2:]
    fmt = fmt.replace('\n', '\n').replace('%g', '%g')
    try:
        msg = fmt % tuple(args) if args else fmt
    except (TypeError, ValueError):
        msg = fmt + ' ' + ' '.join(map(str, args))
    (log.warning if always else log.info)(msg)
