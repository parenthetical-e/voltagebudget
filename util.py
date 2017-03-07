from __future__ import division

from fakespikes import neurons, util, rates
import numpy as np
from voltagebudget.neurons import adex
from voltagebudget.neurons import lif


def k_spikes(t, k, w, dt=1e-3, t_pad=0.1, a=100, seed=42):
    """Generate approx. `k` spikes in the window `w` at time `t`"""

    # Generate a rate pulse
    times = util.create_times(t + t_pad, dt)
    nrns = neurons.Spikes(k, t + t_pad, dt=dt, seed=seed)
    pulse = rates.square_pulse(times, a, t, w, dt, min_a=0)

    # Poisson sample the rate timecourse
    ns, ts = util.to_spiketimes(times, nrns.poisson(pulse))

    return ns, ts


def mean_budget(times, vs, window):
    t0 = window[0]
    tn = window[1]
    m = np.logical_and(times >= t0, times <= tn)

    vs_m = {}
    for k, v in vs.items():
        try:
            len(vs[k])  # error on scalar/float

            vs_m[k] = vs[k][:, m].mean()
        except TypeError:
            vs_m[k] = vs[k]  # copy over scalar values

    return vs_m


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
