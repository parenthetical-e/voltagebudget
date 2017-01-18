from fakespikes import neurons, util, rates


def k_spikes(t, k, w, dt=1e-3, t_pad=0.1, a=100, seed=42):
    """Generate approx. `k` spikes in the window `w` at time `t`"""

    # Generate a rate pulse
    times = util.create_times(t + t_pad, dt)
    nrns = neurons.Spikes(k, t + t_pad, dt=dt, seed=seed)
    pulse = rates.square_pulse(times, a, t, w, dt)

    # Poisson sample the rate timecourse
    ns, ts = util.to_spiketimes(times, nrns.poisson(pulse))

    return ns, ts
