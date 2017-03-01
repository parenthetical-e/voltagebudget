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


def get_budget(t, times, free, osc, comp):

    if times.shape[0] != free.shape[1]:
        raise ValueError("times be the same length as the ncol in all"
                         "other variables (besides t).")

    if (free.shape != osc.shape) or (free.shape != comp.shape):
        raise ValueError("Shape mismatch in voltage varialbles")

    ind = (np.abs(times - t)).argmin()

    return free[:, ind], osc[:, ind], comp[:, ind]


def estimate_communication(t_stim, ns, ts, time=0.3, N=100, t0=2e-3, tn=50e-3):

    _, ts_z = lif(time,
                  N,
                  ns,
                  ts,
                  w_in=(0.2e-9, 0.2e-9),
                  bias=(5e-3, 5e-3 / 5),
                  r_b=0,
                  f=0,
                  A=0,
                  refractory=time,
                  budget=False,
                  report=None)

    # Window for opt analysis
    t0 = t_stim + t0
    tn = t_stim + tn

    # Est communication
    m = np.logical_or(t0 <= ts_z, ts_z <= tn)

    C = 0
    if ts_z[m].size > 0:
        C = ts_z[m].size / float(N)

    return C


def estimate_computation(ns, ts, t0=2e-3, tn=50e-3):
    m = np.logical_or(t0 <= ts, ts <= tn)

    return ts[m].std()
