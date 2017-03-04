from __future__ import division

from fakespikes import neurons, util, rates
import numpy as np
from voltagebudget.neurons import adex
from voltagebudget.neurons import lif


def create_times(t, dt):
    n_steps = int(t * (1.0 / dt))
    times = np.linspace(0, t, n_steps)

    return times


def k_spikes(t, k, w, dt=1e-3, t_pad=0.1, a=100, seed=42):
    """Generate approx. `k` spikes in the window `w` at time `t`"""

    # Generate a rate pulse
    times = util.create_times(t + t_pad, dt)
    nrns = neurons.Spikes(k, t + t_pad, dt=dt, seed=seed)
    pulse = rates.square_pulse(times, a, t, w, dt, min_a=0)

    # Poisson sample the rate timecourse
    ns, ts = util.to_spiketimes(times, nrns.poisson(pulse))

    return ns, ts


def get_budget(t, times, free, osc, comp):

    if times.shape[0] != free.shape[1]:
        raise ValueError("times be the same length as the ncol in all"
                         "other variables (besides t).")

    if (free.shape != osc.shape) or (free.shape != comp.shape):
        raise ValueError("Shape mismatch in voltage varialbles")

    ind = (np.abs(times - t)).argmin()

    return free[:, ind], osc[:, ind], comp[:, ind]


def estimate_communication(t0,
                           tn,
                           ns,
                           ts,
                           coincidence_t=1e-3,
                           coincidence_n=20,
                           time_step=1e-4):

    # Select analysis window 
    m = np.logical_or(t0 <= ts, ts <= tn)
    ts = ts[m]
    ns = ns[m]

    # Create time
    times = create_times(np.max(ts), time_step)

    # Calculate C for every possible coincidence (CC) window, for all time
    Cs = []
    for t in times:
        # Get CC window
        cc0 = t
        ccn = t + coincidence_t
        m = np.logical_or(cc0 <= ts, ts <= ccn)

        # Count spikes in the window
        C_t = 0
        if ts[m].size > 0:
            n_spikes = ts[m].size
            C_t = min(n_spikes / coincidence_n, 1.0)

        Cs.append(C_t)

    # Find largest C
    C = np.max(Cs)

    return C


def estimate_computation(ns, ts, t0=2e-3, tn=50e-3):
    m = np.logical_or(t0 <= ts, ts <= tn)

    return ts[m].std()
