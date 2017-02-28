import numpy as np
from brian2 import *


def lif(time,
        N,
        ns,
        ts,
        w_in=0.3e-9,
        bias=5e-3,
        f=0,
        A=1e-3,
        phi=0,
        r_b=40,
        time_step=1e-4,
        refractory=2e-3,
        budget=True,
        report='text'):
    """Create LIF 'computing' neurons"""

    defaultclock.dt = time_step * second
    prefs.codegen.target = 'numpy'

    if ns.shape[0] == 0:
        return np.array([]), np.array([]), dict()

    try:
        if len(w_in) == 2:
            w_sigma = w_in[1]
            w_in = w_in[0]
    except TypeError:
        w_sigma = 0.0

    try:
        if len(bias) == 2:
            bias_sigma = bias[1]
            bias = bias[0]
    except TypeError:
        bias_sigma = 0.0

    # -----------------------------------------------------------------
    g_l = 10e-9 * siemens

    # comp
    w_in = w_in * siemens
    w_in = w_in / g_l

    w_sigma = w_sigma * siemens
    w_sigma = w_sigma / g_l

    # noise
    w_e = 4e-9 * siemens
    w_e = w_e / g_l

    w_i = 16e-9 * siemens
    w_i = w_i / g_l

    # leak
    g_l = g_l / g_l

    # osc injection
    f *= Hz
    A *= volt
    phi *= second

    # Fixed params
    Et = -54 * mvolt
    Er = -65 * mvolt

    Ee = 0 * mvolt
    Ei = -80 * mvolt

    tau_m = 10 * ms
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
    lif = """
    dv/dt = (g_l * (Er - v) + I_syn + I_osc + I) / tau_m : volt
    I_syn = g_in * (Ee - v) + g_e * (Ee - v) + g_i * (Ei - v) : volt
    dg_in/dt = -g_in / tau_ampa : 1
    dg_e/dt = -g_e / tau_ampa : 1
    dg_i/dt = -g_i / tau_gaba : 1
    I_osc = A * sin((t + phi) * f * 2 * pi) : volt
    I : volt
    """

    # Define the neurons
    P_e = NeuronGroup(
        N,
        lif,
        threshold='v > Et',
        reset='v = Er',
        refractory=refractory * second,
        method='euler')

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
    P_stim = SpikeGeneratorGroup(np.max(ns) + 1, ns, ts * second)
    C_stim = Synapses(P_stim, P_e, model='w : 1', on_pre='g_in += w')
    C_stim.connect()
    C_stim.w = 'clip(w_in + ((j+1)/(j+1) * w_sigma * randn()), 0.0, 1e6)'

    # -----------------------------------------------------------------
    # Deinfe variable
    spikes_e = SpikeMonitor(P_e)
    traces_e = StateMonitor(P_e, ['v'], record=True)

    # Define basic net
    net = Network(P_be, P_bi, P_e, C_be, C_bi, traces_e)
    net.store('no_stim')

    # If budgets are desired, run the net without
    # any stimululation. (This strictly speaking isn't
    # necessary, but I can't get Brian to express the needed
    # diff eq to get the osc budget term in one pass.)
    if budget:
        from copy import deepcopy
        net.run(time * second, report=report)
        v_osc = deepcopy(traces_e.v_)

    net.restore('no_stim')
    net.add([P_stim, C_stim, spikes_e])
    net.run(time * second, report=report)

    # Extract spikes
    ns_e = spikes_e.i_
    ts_e = spikes_e.t_
    result = [ns_e, ts_e]

    if budget:
        vm = traces_e.v_
        v_comp = (vm - v_osc) + np.mean(v_osc)
        v_free = vm - float(Er)
        v_b = float(Et - Er)
        vs = dict(
            vm=vm,
            comp=v_comp,
            osc=v_osc,
            free=v_free,
            budget=v_b,
            rest=float(Er))

        result.append(vs)

    return result


def adex(time,
         N,
         ns,
         ts,
         a=(-1.0e-9, 1.0e-9),
         b=(10e-12, 60.0e-12),
         Ereset=-48e-3,
         w_in=0.8e-9,
         bias=0.5e-9,
         f=0,
         A=1e-3,
         phi=0,
         r_b=40,
         time_step=1e-4,
         budget=True,
         report='text'):
    """Create AdEx 'computing' neurons"""

    defaultclock.dt = time_step * second
    prefs.codegen.target = 'numpy'

    if ns.shape[0] == 0:
        return np.array([]), np.array([]), dict()

    try:
        if len(w_in) == 2:
            w_sigma = w_in[1]
            w_in = w_in[0]
    except TypeError:
        w_sigma = 0.0

    try:
        if len(bias) == 2:
            bias_sigma = bias[1]
            bias = bias[0]
    except TypeError:
        bias_sigma = 0.0

    # -----------------------------------------------------------------
    C = 281 * pF
    g_l = 30 * nS

    # comp
    w_in = w_in * siemens
    w_sigma = w_sigma * siemens

    # noise
    w_e = 4e-9 * siemens
    w_i = 16e-9 * siemens

    # osc injection
    f *= Hz
    A *= amp
    phi *= second

    # neuron kinetics
    El = -70.6 * mV
    Et = -50.4 * mV
    delta_t = 2 * mV
    tau_w = 40 * ms
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
    I_osc = A * sin((t + phi) * f * 2 * pi) : amp
    Er : volt
    a: siemens
    b: amp
    I : amp
    """

    P_e = NeuronGroup(
        N,
        model=eqs,
        threshold='v > Ecut',
        reset="v = Er; w += b",
        method='euler')

    P_e.v = El

    # Set random (?) recovery physics?
    try:
        P_e.a = np.random.uniform(a[0], a[1], N) * siemens
    except TypeError:
        P_e.a = a * siemens
    try:
        P_e.b = np.random.uniform(b[0], b[1], N) * amp
    except TypeError:
        P_e.b = b * amp
    try:
        P_e.Er = np.random.uniform(Ereset[0], Ereset[1], N) * volt
    except TypeError:
        P_e.Er = Ereset * volt

    P_e.w = P_e.a * (P_e.v - El)

    # Random bias?
    Is = bias + (bias_sigma * np.random.normal(0, 1, len(P_e)))
    P_e.I = Is * amp

    # Set up the 'network'
    # Noise
    C_be = Synapses(P_be, P_e, on_pre='g_e += w_e')
    C_be.connect('i == j')
    C_bi = Synapses(P_bi, P_e, on_pre='g_i += w_i')
    C_bi.connect('i == j')

    # Stim
    P_stim = SpikeGeneratorGroup(np.max(ns) + 1, ns, ts * second)
    C_stim = Synapses(P_stim, P_e, on_pre='g_in += w_in')
    C_stim.connect()

    # -----------------------------------------------------------------
    # Deinfe variable
    spikes_e = SpikeMonitor(P_e)
    traces_e = StateMonitor(P_e, ['v'], record=True)

    # Define basic net
    net = Network(P_be, P_bi, P_e, C_be, C_bi, traces_e)
    net.store('no_stim')

    # If budgets are desired, run the net without
    # any stimululation. (This strictly speaking isn't
    # necessary, but I can't get Brian to express the needed
    # diff eq to get the osc budget term in one pass.)
    if budget:
        from copy import deepcopy
        net.run(time * second, report=report)
        v_osc = deepcopy(traces_e.v_)

    net.restore('no_stim')
    net.add([P_stim, C_stim, spikes_e])
    net.run(time * second, report=report)

    # Extract spikes
    ns_e = spikes_e.i_
    ts_e = spikes_e.t_
    result = [ns_e, ts_e]

    if budget:
        vm = traces_e.v_
        v_comp = (vm - v_osc) + np.mean(v_osc)
        v_free = vm - float(Et)
        v_b = Et - Er
        vs = dict(vm=vm, comp=v_comp, osc=v_osc, free=v_free, budget=v_b)

        result.append(vs)

    return result
