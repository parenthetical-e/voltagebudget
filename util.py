from __future__ import division

from fakespikes import neurons, rates
from fakespikes import util as fsutil

import numpy as np
from voltagebudget.neurons import adex
from voltagebudget.neurons import lif


def k_spikes(t, k, w, dt=1e-3, t_pad=0.1, a=100, seed=42):
    """Generate approx. `k` spikes in the window `w` at time `t`"""

    # Generate a rate pulse
    times = fsutil.create_times(t + t_pad, dt)
    nrns = neurons.Spikes(k, t + t_pad, dt=dt, seed=seed)
    pulse = rates.square_pulse(times, a, t, w, dt, min_a=0)

    # Poisson sample the rate timecourse
    ns, ts = fsutil.to_spiketimes(times, nrns.poisson(pulse))

    return ns, ts


def filter_times(times, window):
    m = np.logical_and(times >= window[0], times <= window[1])
    return times[m]


def filter_spikes(ts, ns, window):
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


def mean_budget(times, N, ns, ts, vs, window, spiked_only=True):
    if ns.size == 0:
        raise ValueError("(ns, ts) are empty")

    n_idx = range(N)
    if spiked_only:
        n_idx = sorted(np.unique(ns))

    from collections import defaultdict

    vs_m = defaultdict(list)
    for n, t in zip(ns, ts):
        # Find first passage, t*
        w = (t + window[0], t + window[1])
        vs_f = filter_budget(times, vs, w)

        for k, v in vs_f.items():
            try:
                len(v)  # error on scalar/float

                vs_m[k] = v[n, :].mean()
            except TypeError:
                vs_m[k] = v  # copy over scalar values

    for k, v in vs_m.items():
        vs_m[k] = np.mean(v)

    print(vs_m)

    return vs_m


def estimate_rate(ns, ts, window):
    t0 = window[0]
    tn = window[1]
    m = np.logical_and(t0 <= ts, ts <= tn)

    return len(ts[m])


def estimate_communication(times,
                           ns,
                           ts,
                           window,
                           coincidence_t=1e-3,
                           coincidence_n=20,
                           time_step=1e-4):

    # Define overall analysis window 
    t0 = window[0]
    tn = window[1]
    if tn + coincidence_t > times.max():
        raise ValueError("Final window must be less than max value in times")

    m = np.logical_and(t0 <= ts, ts <= tn)
    ts = ts[m]
    ns = ns[m]

    # Calculate C for every possible coincidence (CC) window, for all time
    Cs = []
    for t in times:
        # Get CC window
        cc0 = t
        ccn = t + coincidence_t
        m = np.logical_and(cc0 <= ts, ts <= ccn)

        # Count spikes in the window
        C_t = 0
        if ts[m].size > 0:
            n_spikes = ts[m].size
            C_t = min(n_spikes / coincidence_n, 1.0)

        Cs.append(C_t)

    # Find largest C
    C = np.max(Cs)

    return C


def estimate_computation(times, ns, ts, window):
    t0 = window[0]
    tn = window[1]
    m = np.logical_and(t0 <= ts, ts <= tn)

    ts = ts[m]

    if ts.size > 0:
        return ts.std()
    else:
        return 0.0
