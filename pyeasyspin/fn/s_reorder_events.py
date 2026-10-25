"""Reorder events of a pulse sequence (moving pulses) into a strictly sequential time line."""
import numpy as np

from .._registry import nout
from .._mat import vec


def _mround(x):
    """MATLAB round (half away from zero), elementwise."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


@nout(2)
def s_reorder_events(EventLengths=None, PulseList=None):
    """(NewSequence, NewEventLengths): event indices (1-based, 1 x nEvents) in time order and the
    lengths of the resulting intervals, rounded to 1e-10 us. Overlapping pulses raise
    'Pulse Overlap! Events: ...'; a pulse squeezed to zero length is swapped with its successor."""
    EventLengths = np.asarray(EventLengths, dtype=float)
    shape = EventLengths.shape if EventLengths.ndim == 2 else (1, EventLengths.size)
    EventLengths = vec(EventLengths)
    PulseList = vec(np.asarray(PulseList, dtype=float))
    nEvents = EventLengths.size
    t = np.concatenate([[0.0], np.cumsum(EventLengths)])
    order = np.argsort(t, kind='stable')                  # MATLAB sort is stable
    TimeIntervals = t[order]
    NewSequence = (order + 1).astype(float)[:-1]
    NewEventLengths = np.zeros(nEvents)
    for k in range(nEvents):
        NewEventLengths[k] = TimeIntervals[k + 1] - TimeIntervals[k]
    num_dig = 10
    NewEventLengths = _mround(NewEventLengths * (10 ** num_dig)) / (10 ** num_dig)

    EventMatrix = np.zeros((nEvents, nEvents))
    for iEvent in range(nEvents):
        t_event = [t[iEvent], t[iEvent] + EventLengths[iEvent]]
        tEventStart, tEventEnd = min(t_event), max(t_event)
        idx = (TimeIntervals >= tEventStart) & (TimeIntervals < tEventEnd)
        # EventMatrix(iEvent,idx) = ... with a logical index longer than the row: MATLAB grows the
        # matrix to nEvents+1 columns when the last entry is true
        if idx.size > EventMatrix.shape[1] and idx[-1]:
            EventMatrix = np.concatenate([EventMatrix, np.zeros((nEvents, idx.size - EventMatrix.shape[1]))], axis=1)
        cols = np.nonzero(idx)[0]
        EventMatrix[iEvent, cols] = 2 if PulseList[iEvent] == 1 else 1
    sums = np.sum(EventMatrix == 2, axis=0)
    if np.any(sums > 1):
        Overlaps = np.nonzero(sums > 1)[0]
        OverlappingEvents = np.sum(EventMatrix[:, Overlaps] == 2, axis=1)
        OverlappingEvents = np.nonzero(OverlappingEvents > 0)[0] + 1
        msg = 'Pulse Overlap! Events: ' + _num2str(OverlappingEvents[0]) + ', ' + _num2str(OverlappingEvents[1])
        if len(Overlaps) > 2:
            for i in range(3, len(Overlaps) + 1):
                msg = msg + ', ' + _num2str(OverlappingEvents[i - 1])
        raise ValueError(msg)
    for iEvent in range(nEvents):
        if PulseList[int(NewSequence[iEvent]) - 1] and NewEventLengths[iEvent] == 0:
            if iEvent + 1 >= nEvents:
                raise IndexError('Index exceeds the number of array elements (%d).' % nEvents)
            ShortenedPulse = NewSequence[iEvent]
            NewSequence[iEvent] = NewSequence[iEvent + 1]
            NewSequence[iEvent + 1] = ShortenedPulse
    return NewSequence.reshape(1, -1), NewEventLengths.reshape(shape, order='F')


def _num2str(v):
    return str(int(v))
