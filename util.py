from __future__ import division

from fakespikes import neurons, rates
from fakespikes import util as fsutil

import numpy as np
from copy import deepcopy


def filter_spikes(ns, ts, window):
    m = np.logical_and(ts >= window[0], ts <= window[1])

    return ns[m], ts[m]


def filter_budget(times, vs, window):
    m = np.logical_and(times >= window[0], times <= window[1])

    filtered = {}
    for k, v in vs.items():
        try:
            len(vs[k])  # error on scalar/float

            filtered[k] = vs[k][:, m]
        except TypeError:
            filtered[k] = vs[k]  # copy over scalar values

    return filtered
