import os
import json
import csv
import numpy as np

import voltagebudget
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


def deviants(ts, percent_change, dt, argfn=np.argmax):
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
    deltas = np.absolute(ts - np.mean(ts))

    # Opt!
    ts_opt = ts.copy()

    while adjusted > target:
        # Find largest delta 
        k = argfn(deltas)

        # and shift that spike toward the mean
        if ts_opt[k] < 0:
            ts_opt[k] += dt
        elif ts_opt[k] > 0:
            ts_opt[k] -= dt
        else:
            pass

        # Recalc the deltas
        deltas = np.absolute(ts_opt - np.mean(ts_opt))

        # Update rolling MAD
        adjusted = mad(ts_opt)

    return initial, target, np.asarray(ts_opt)
