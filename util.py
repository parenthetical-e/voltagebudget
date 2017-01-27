from fakespikes import neurons, util, rates
import numpy as np


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
