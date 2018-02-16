import os
import json
import csv
import numpy as np

import voltagebudget
from copy import deepcopy
from voltagebudget.util import mad


def _create_target(ts, percent_change):
    initial = mad(ts)
    target = initial - (initial * percent_change)
    return initial, target


def uniform(ts, percent_change):
    if not (0 <= percent_change <= 1):
        raise ValueError("p must be between (0-1).")

    if ts.size == 0:
        return ts

    ts = np.asarray(ts)

    # -
    initial, target = _create_target(ts, percent_change)

    dt = np.absolute(initial - target)

    ts_opt = []
    M = np.mean(ts)
    for t in ts:
        if t < M:
            t += dt
        elif t > M:
            t -= dt
        else:
            t += 0

        ts_opt.append(t)
    ts_opt = np.asarray(ts_opt)

    obs = mad(ts_opt)

    return initial, target, obs, ts_opt


def _delta(ts):
    return np.absolute(ts - np.mean(ts))


def _diffs(ts):
    ts = np.sort(ts)

    t_last = ts[0]
    ds = []
    for t in ts[1:]:
        ds.append(t_last - t)
        t_last = deepcopy(t)

    return np.asarray(ds)


def coincidence(ts, percent_change):
    """Coordinate by increasing coincidences"""

    if not (0 <= percent_change <= 1):
        raise ValueError("p must be between (0-1).")
    if ts.size == 0:
        return ts

    # Prelim...
    ts = np.sort(ts)

    # Figure out initial and target MAD.
    initial, target = _create_target(ts, percent_change)

    # Est the seq. differences between ts,
    # and index their order.
    i = np.argmin(_diffs(ts))

    # -
    # Shift closest spikes (smallest deltas)
    # to be the same until target MAD is achieved.
    ts_opt = ts.copy()
    adjusted = initial
    while adjusted > target:
        ts_opt[i + 1] = ts_opt[i]

        adjusted = mad(ts_opt)

        i = np.argmin(_diffs(ts_opt))

    return initial, target, adjusted, np.asarray(ts_opt)


def max_deviant(ts, percent_change, dt=0.1e-3):
    if dt < 0:
        raise ValueError("dt must be positive.")
    if not (0 <= percent_change <= 1):
        raise ValueError("p must be between (0-1).")
    if ts.size == 0:
        return ts

    ts = np.asarray(ts)

    # -
    initial, target = _create_target(ts, percent_change)

    # We'll always want the biggest....
    deltas = _delta(ts)

    # Init
    ts_opt = ts.copy()
    adjusted = initial

    # -
    while adjusted > target:
        # Find most extreme delta 
        k = np.argmax(deltas)

        # and shift that spike toward the mean
        M = np.mean(ts_opt)
        if ts_opt[k] < M:
            ts_opt[k] += dt
        elif ts_opt[k] > M:
            ts_opt[k] -= dt
        else:
            ts_opt[k] += 0.0

        # Recalc the deltas
        deltas = _delta(ts_opt)

        # Update rolling MAD
        adjusted = mad(ts_opt)

    return initial, target, adjusted, np.asarray(ts_opt)
