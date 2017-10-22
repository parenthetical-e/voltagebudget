import numpy as np
from brian2 import *


def shadow_adex(time, ns, ts, **adex_kwargs):
    """Est. the 'shadow voltage' of the AdEx membrane voltage."""
    # In the neuron can't fire, we're in the shadow realm!
    Et = 1000  # 1000 volts is infinity, for neurons.
    _, _, budget = adex(time, ns, ts, budget=True, Et=Et, **adex_kwargs)

    return budget['vm'].flatten(), budget


def adex(time,
         ns,
         ts,
         a=0e-9,
         b=10e-12,
         tau_w=30e-3,
         tau_m=20e-3,
         Erheo=-48e-3,
         delta_t=2e-3,
         w_in=0.8e-9,
         tau_in=5e-3,
         bias=0.5e-9,
         Et=-50.4e-3,
         f=0,
         A=1e-3,
         phi=0,
         time_step=1e-4,
         budget=True,
         report='text',
         seed=None):
    """A AdEx neuron"""
    np.random.seed(seed)
    defaultclock.dt = time_step * second
    prefs.codegen.target = 'numpy'

    # -----------------------------------------------------------------
    # If there's no input, return empty 
    if ns.shape[0] == 0:
        return np.array([]), np.array([]), dict()

    # -----------------------------------------------------------------
    # tau_m:
    tau_m *= second
    g_l = 30 * nS
    C = tau_m * g_l

    # Other neuron params
    El = -70.6 * mV
    Et *= volt
    Erheo *= volt

    # Comp vars
    w_in *= siemens
    tau_in *= second
    bias *= amp
    a *= siemens
    b *= amp
    delta_t *= volt
    tau_w *= second
    Ecut = Et + 5 * delta_t  # practical threshold condition

    # osc injection
    f *= Hz
    A *= amp
    phi *= second

    # -----------------------------------------------------------------
    # Define neuron and its connections
    eqs = """
    dv/dt = (g_l * (El - v) + g_l * delta_t * exp((v - Et) / delta_t) + I_in + I_osc + bias - w) / C : volt
    dw/dt = (a * (v - El) - w) / tau_w : amp
    I_in = g_in * (v - El) : amp
    dg_in/dt = -g_in / tau_in : siemens
    I_osc = A * sin((t + phi) * f * 2 * pi) : amp
    """

    P_e = NeuronGroup(
        1,
        model=eqs,
        threshold='v > Ecut',
        reset="v = Erheo; w += b",
        method='euler')

    # Init
    P_e.v = El
    P_e.w = a * (P_e.v - El)

    # Set up the 'network'
    P_stim = SpikeGeneratorGroup(np.max(ns) + 1, ns, ts * second)
    C_stim = Synapses(P_stim, P_e, on_pre='g_in += w_in')
    C_stim.connect()

    # -----------------------------------------------------------------
    # Deinfe variable
    spikes_e = SpikeMonitor(P_e)
    traces_e = StateMonitor(P_e, ['v'], record=True)

    # Define basic net
    net = Network(P_e, traces_e)
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
    ns_e = np.asarray(spikes_e.i_)
    ts_e = np.asarray(spikes_e.t_)
    result = [ns_e, ts_e]

    if budget:
        times = np.asarray(traces_e.t_)

        Ecut = float(Ecut)
        El = float(El)

        v_b = float(Ecut - El)
        vm = traces_e.v_

        v_comp = (vm - v_osc) + np.mean(v_osc) - float(El)
        v_osc = v_osc - float(El)
        v_free = float(Ecut) - vm

        vs = dict(
            times=times,
            vm=vm,
            comp=v_comp,
            osc=v_osc,
            free=v_free,
            budget=v_b,
            rest=El,
            tau_m=float(C / g_l))

        result.append(vs)

    return result
