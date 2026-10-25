"""Format the time between two clock vectors."""
import datetime
import math
from .._registry import nout


def _todt(c):
    if isinstance(c, datetime.datetime):
        return c
    c = [float(x) for x in (c.ravel() if hasattr(c, 'ravel') else c)]
    sec = int(c[5]); us = int(round((c[5] - sec) * 1e6))
    return datetime.datetime(int(c[0]), int(c[1]), int(c[2]), int(c[3]), int(c[4]), sec) + datetime.timedelta(microseconds=us)


@nout(4)
def elapsedtime(startTime=None, endTime=None):
    """(hmsString, Hours, Minutes, Seconds) between two MATLAB clock vectors (or datetimes)."""
    Time = (_todt(endTime) - _todt(startTime)).total_seconds() / 3600
    Hours = math.trunc(Time)
    rem = (Time - Hours) * 60
    Minutes = math.trunc(rem)
    Seconds = (rem - Minutes) * 60
    if Hours > 0:
        s = '%dh%dm%0.3fs' % (Hours, Minutes, Seconds)
    elif Minutes > 0:
        s = '%dm%0.3fs' % (Minutes, Seconds)
    else:
        s = '%0.3fs' % Seconds
    return s, Hours, Minutes, Seconds
