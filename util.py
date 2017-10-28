from __future__ import division

from fakespikes import neurons, rates
from fakespikes import util as fsutil

import numpy as np
from copy import deepcopy


def poisson_impulse(t, t_stim, w, rate, n, dt=1e-3, seed=None):
    """Create a pulse of spikes w seconds wide, starting at t_stim."""

    # Poisson sample the rate over w
    times = fsutil.create_times(t, dt)
    nrns = neurons.Spikes(n, t, dt=dt, seed=seed)
    pulse = rates.square_pulse(times, rate, t_stim, w, dt, min_a=0)

    ns, ts = fsutil.to_spiketimes(times, nrns.poisson(pulse))

    return ns, ts


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
        except IndexError:
            filtered[k] = vs[k][m]
        except TypeError:
            filtered[k] = vs[k]  # copy over scalar values

    return filtered
