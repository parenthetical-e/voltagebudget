from __future__ import division

from fakespikes import neurons, rates
from fakespikes import util as fsutil

import numpy as np
from copy import deepcopy


def poisson_impulse(t, t_stim, w, rate, n, dt=1e-3, seed=None):
    """Create a pulse of spikes w seconds wide, starting at t_stim"""

    # Poisson sample the rate over w
    times = fsutil.create_times(t, dt)
    nrns = neurons.Spikes(n, t, dt=dt, seed=seed)
    pulse = rates.square_pulse(times, rate, t_stim, w, dt, min_a=0)

    ns, ts = fsutil.to_spiketimes(times, nrns.poisson(pulse))

    return ns, ts


def k_spikes(t,
             k,
             w,
             dt=1e-3,
             t_pad=0.1,
             a0=100,
             n=10,
             a_step=.1,
             max_iterations=1000,
             seed=42):
    """Generate `k` spikes in the window `w` at time `t`"""

    # Generate a rate pulse
    times = fsutil.create_times(t + t_pad, dt)
    nrns = neurons.Spikes(n, t + t_pad, dt=dt, seed=seed)

    i = 0
    k0 = 0
    a = deepcopy(a0)
    while k != k0:

        pulse = rates.square_pulse(times, a, t, w, dt, min_a=0)

        # Poisson sample the rate timecourse
        ns, ts = fsutil.to_spiketimes(times, nrns.poisson(pulse))

        k0 = len(ns)
        if k0 < k:
            a += a_step
        if k0 > k:
            a -= a_step

        i += 1
        if i > max_iterations:
            print(">>> Convergence FAILED!")
            break

    print(">>> {} spikes generated after {} iterations.".format(k0, i))

    return ns, ts


def filter_times(times, window):
    m = np.logical_and(times >= window[0], times <= window[1])
    return times[m]


def filter_spikes(ns, ts, window):
    m = np.logical_and(ts >= window[0], ts <= window[1])

    return ns[m], ts[m]


def l2(va, vb):
    """Calculates the L2 norm between two voltage series."""

    return np.trapz((va - vb)**2)


def levenshtein(a, b):
    """Calculates the Levenshtein distance between a and b.

    Note: a and b are two sequences
    """

    a = list(a)
    b = list(b)
    n, m = len(a), len(b)

    # Make sure n <= m, to use O(min(n,m)) space
    if n > m:
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[-1]


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
        # Find first passage, t
        w = (t + window[0], t + window[1])
        vs_f = filter_budget(times, vs, w)

        # Gather voltages for each passage
        for k, v in vs_f.items():
            try:
                len(v)  # error on scalar/float

                vs_m[k].append(v[n, :].mean())
            except TypeError:
                vs_m[k] = v  # copy over scalar values

    # Avg voltages for all neurons
    for k, v in vs_m.items():
        vs_m[k] = np.mean(v)

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
                           return_all=False,
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

    # Find avg C
    C = np.mean(Cs)
    out = C

    if return_all:
        out = (C, Cs)

    return out


def estimate_computation(times, ns, ts, window):
    print("Should you be using me?")

    t0 = window[0]
    tn = window[1]
    m = np.logical_and(t0 <= ts, ts <= tn)

    ts = ts[m]

    if ts.size > 0:
        return ts.std()
    else:
        return 0.0
