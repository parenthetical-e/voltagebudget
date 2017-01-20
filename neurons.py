import numpy as np
from brian2 import *


def lif(time,
        N,
        k,
        ns,
        ts,
        w_in=0.3e-9,
        bias=5e-3,
        f=0,
        A=1e-3,
        r_b=40,
        time_step=1e-4,
        sigma_scale=10.0,
        report='text'):
    """Create LIF 'computing' neurons"""
    # -----------------------------------------------------------------
    g_l = 10e-9 * siemens

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
    w_e = 4e-9 * siemens
    w_i = 16e-9 * siemens

    w_e = w_e / g_l
    w_i = w_i / g_l
    g_l = g_l / g_l

    # osc injection
    f *= Hz
    A *= volt

    # Fixed params
    Et = -54 * mvolt
    Er = -65 * mvolt

    Ee = 0 * mvolt
    Ei = -80 * mvolt

    tau_m = 10 * ms
    tau_ampa = 5e-3 * second
    tau_gaba = 10e-3 * second

    time_step *= second
    defaultclock.dt = time_step

    # -----------------------------------------------------------------
    # E/I noise
    Nb = 1000
    r_e = r_b * Hz
    r_i = r_b * Hz
    P_be = PoissonGroup(Nb, r_e)
    P_bi = PoissonGroup(Nb, r_i)

    # Define neuron and its connections
    lif = """
    dv/dt = (g_l * (Er - v) + I_syn + I_osc + I) / tau_m : volt
    I_syn = g_in * (Ee - v) + g_e * (Ee - v) + g_i * (Ei - v) : volt
    dg_in/dt = -g_in / tau_ampa : 1
    dg_e/dt = -g_e / tau_ampa : 1
    dg_i/dt = -g_i / tau_gaba : 1
    I_osc = A * sin(t * f * 2 * pi) : volt
    I : volt
    """

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


def adex(time,
         N,
         k,
         ns,
         ts,
         Ereset=48e-3,
         w_in=0.3e-9,
         bias=5e-5,
         f=0,
         A=1e-3,
         r_b=40,
         time_step=0.01e-3,
         sigma_scale=20.0,
         report='text'):
    """Create LIF 'computing' neurons"""
    defaultclock.dt = time_step * second

    C = 281 * pF
    g_l = 30 * nS

    w_in = w_in * siemens
    # w_in = w_in / g_l

    if np.allclose(sigma_scale, 0.0):
        Er_sigma = 0.0
    else:
        Er_sigma = Ereset / sigma_scale

    # noise
    w_e = 4e-9 * siemens
    w_i = 16e-9 * siemens

    # w_e = w_e / g_l
    # w_i = w_i / g_l
    # g_l = g_l / g_l

    # osc injection
    f *= Hz
    A *= amp

    # neuron kinetics
    El = -70.6 * mV
    Et = -50.4 * mV
    delta_t = 2 * mV
    tau_w = 40 * ms
    a = 4 * nS
    b = 0.08 * nA
    I = .8 * nA
    Ecut = Et + 5 * delta_t  # practical threshold condition

    # background synapses
    Ee = 0 * mvolt
    Ei = -80 * mvolt
    tau_ampa = 5e-3 * second
    tau_gaba = 10e-3 * second

    # -----------------------------------------------------------------
    # E/I noise
    Nb = 1000
    r_e = r_b * Hz
    r_i = r_b * Hz
    P_be = PoissonGroup(Nb, r_e)
    P_bi = PoissonGroup(Nb, r_i)

    # Define neuron and its connections
    eqs = """
    dv/dt = (g_l * (El - v) + g_l * delta_t * exp((v - Et) / delta_t) + I_syn + I_osc + I - w) / C : volt
    dw/dt = (a * (v - El) - w) / tau_w : amp
    I_syn = g_in * (Ee - v) + g_e * (Ee - v) + g_i * (Ei - v) : amp
    dg_in/dt = -g_in / tau_ampa : siemens
    dg_e/dt = -g_e / tau_ampa : siemens
    dg_i/dt = -g_i / tau_gaba : siemens
    I_osc = A * sin(t * f * 2 * pi) : amp
    Er : volt
    I : amp
    """

    P_e = NeuronGroup(
        N,
        model=eqs,
        threshold='v > Ecut',
        reset="v = Er; w += b",
        method='euler')

    P_e.v = El
    P_e.w = a * (P_e.v - El)
    P_e.Er = (-Ereset + (Er_sigma * np.random.normal(0, 1, len(P_e)))) * volt
    P_e.I = bias * amp

    # Set up the 'network'
    # Noise
    C_be = Synapses(P_be, P_e, on_pre='g_e += w_e')
    C_be.connect('i == j')
    C_bi = Synapses(P_bi, P_e, on_pre='g_i += w_i')
    C_bi.connect('i == j')

    # Stim
    P_stim = SpikeGeneratorGroup(k, ns, ts * second)
    C_stim = Synapses(P_stim, P_e, on_pre='g_in += w_in')
    C_stim.connect()

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