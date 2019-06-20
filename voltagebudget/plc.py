import os
import json
import csv
import numpy as np

import voltagebudget
from copy import deepcopy
from voltagebudget.util import mad

# def _create_target(ts, percent_change):
#     initial = mad(ts)
#     target = initial - (initial * percent_change)
#     return initial, target


def uniform(ts, initial, target):
    """Shift each spike time a uniform amount

    This is the min. max. or optimal policy for individual neuron errors"""

    # Init
    ts = np.asarray(ts)
    if ts.size == 0:
        return ts

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

    adjusted = mad(ts_opt)

    return initial, target, adjusted, ts_opt


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


def _find_value(x, array):
    array = np.asarray(array)
    idx = (np.abs(array - x)).argmin()
    return idx


def coincidence(ts, initial, target, min_distance=1e-6, max_iterations=1000):
    """Coordinate by increasing coincidences.
    
    This is the optimal policy to minimize the minimum
    description length (MDL). 
    
    https://en.wikipedia.org/wiki/Minimum_description_length
    """

    if np.isclose(initial, target):
        return initial, initial, initial, ts

    # Init
    ts = np.asarray(ts)
    if ts.size == 0:
        return ts

    # Prelim...
    ts = np.sort(ts)

    # -
    # Shift closest spikes (smallest deltas)
    # to be the same until target MAD is achieved.
    N = ts.size
    ts_opt = ts.copy()
    adjusted = initial

    n = 0
    while (adjusted >= target) and (n < max_iterations):
        # Find distances
        deltas = _diffs(ts_opt)

        # Must be not already a coincidence
        deltas[deltas > min_distance] = np.nan
        # print(deltas > min_distance)

        # Find smallest distance
        i = np.nanargmin(deltas)

        # Shift
        # Est the seq. differences between ts,
        # and index their order.
        # Note: nans are last in argsort.
        ts_opt[i + 1] = ts_opt[i]

        # Update stats/counters
        adjusted = mad(ts_opt)

        n += 1

    return initial, target, adjusted, np.asarray(ts_opt)


def max_deviant(ts,
                initial,
                target,
                side='both',
                mode_fm=np.mean,
                dt=0.01e-3,
                max_iterations=250000):
    """Coordinate by adjusting the max variance neurons

    This is the optimal policy for the network as a whole.
    """

    # Init
    ts = np.asarray(ts)
    if ts.size == 0:
        return ts

    # Sanity
    if dt < 0:
        raise ValueError("dt must be positive.")

    # -
    adjusted = deepcopy(initial)
    idx = np.argsort(ts)
    ts_opt = ts.copy()[idx]
    M = mode_fm(ts_opt)

    # Which side of ts too look at?
    if side == 'both':
        mask = np.ones_like(ts_opt, dtype=np.bool)
    elif side == 'left':
        mask = ts <= M
    else:
        raise ValueError("side must be, ('both', 'left'")

    deltas = _delta(ts_opt)

    # -
    iter_count = 0
    while adjusted > target:
        # Find index of farthest spike
        k = np.argmax(deltas[mask])

        # and shift that spike toward the mean
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

        # Avoid inf
        iter_count += 1
        if iter_count > max_iterations:
            print(">>> max_deviant stopped at {} iterations".format(
                max_iterations))
            break

    # Re-sort ts_opt to match the order in intial ts
    ts_re = np.zeros_like(ts_opt)
    for n, i in enumerate(idx):
        ts_re[i] = ts_opt[n]

    return initial, target, adjusted, ts_re
