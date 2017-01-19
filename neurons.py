import numpy as np
from brian2 import *


def lif(time,
        N,
        k,
        ns,
        ts,
        w_in,
        bias,
        f=0,
        a=1e-3,
        Nb=1000,
        r_e=40,
        r_i=40,
        w_e=4e-9,
        w_i=16e-9,
        tau_e=5e-3,
        tau_i=10e-3,
        g_l=10e-9,
        time_step=1e-4,
        sigma_scale=10.0,
        report='text'):
    """Create LIF 'computing' neurons"""
    # -----------------------------------------------------------------
    # User params
    g_l = g_l * siemens

    # comp
    w_in = w_in * siemens
    w_in = w_in / g_l

    if np.allclose(sigma_scale, 0.0):
        w_sigma = 0.0
        bias_sigma = 0.0
    else:
        w_sigma = w_in / sigma_scale
        bias_sigma = bias / sigma_scale

    # noise
    w_e = w_e * siemens
    w_i = w_i * siemens

    w_e = w_e / g_l
    w_i = w_i / g_l
    g_l = g_l / g_l

    # osc injection
    f *= Hz
    a *= volt

    r_e = r_e * Hz
    r_i = r_i * Hz

    # Fixed params
    Et = -54 * mvolt
    Er = -65 * mvolt

    Ee = 0 * mvolt
    Ei = -80 * mvolt

    tau_m = 10 * ms
    tau_ampa = tau_e * second
    tau_gaba = tau_i * second

    time_step *= second
    defaultclock.dt = time_step

    # -----------------------------------------------------------------
    # Define neuron and its connections
    lif = """
    dv/dt = (g_l * (Er - v) + I_syn + I_osc + I) / tau_m : volt
    I_syn = g_in * (Ee - v) + g_e * (Ee - v) + g_i * (Ei - v) : volt
    dg_in/dt = -g_in / tau_ampa : 1
    dg_e/dt = -g_e / tau_ampa : 1
    dg_i/dt = -g_i / tau_gaba : 1
    I_osc = a * sin(t * f * 2 * pi) : volt
    I : volt
    """

    # E/I noise
    P_be = PoissonGroup(Nb, r_e)
    P_bi = PoissonGroup(Nb, r_i)

    # Define the neurons
    P_e = NeuronGroup(
        N,
        lif,
        threshold='v > Et',
        reset='v = Er',
        refractory=2 * ms,
        method='rk2')

    P_e.v = Er

    Is = bias + (bias_sigma * np.random.normal(0, 1, len(P_e)))
    P_e.I = Is * volt

    # Set up the 'network'
    # Noise
    C_be = Synapses(P_be, P_e, on_pre='g_e += w_e')
    C_be.connect('i == j')
    C_bi = Synapses(P_bi, P_e, on_pre='g_i += w_i')
    C_bi.connect('i == j')

    # Stim
    P_stim = SpikeGeneratorGroup(k, ns, ts * second)
    C_stim = Synapses(P_stim, P_e, model='w : 1', on_pre='g_in += w')
    C_stim.connect()
    C_stim.w = 'clip(w_in + (j * w_sigma * randn()), 0.0, 1e6)'

    # -----------------------------------------------------------------
    # Run, and extract results
    spikes_e = SpikeMonitor(P_e)
    traces_e = StateMonitor(P_e, ['v', 'g_in', 'I_osc', 'I'], record=True)

    run(time * second, report=report)

    # Extract spikes
    ns_e = spikes_e.i_
    ts_e = spikes_e.t_

    # And the voltages
    vm = traces_e.v_
    v_comp = (traces_e.g_in_ * (float(Ee) - vm) + traces_e.I_)
    v_osc = traces_e.I_osc
    v_free = vm - float(Et)
    vs = dict(vm=vm, comp=v_comp, osc=v_osc, free=v_free)

    return ns_e, ts_e, vs