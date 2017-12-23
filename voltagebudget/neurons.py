import inspect
import csv
import numpy as np
from brian2 import *
from copy import deepcopy
from voltagebudget.util import step_waves


def shadow_adex(N, time, ns, ts, **adex_kwargs):
    """Est. the 'shadow voltage' of the AdEx membrane voltage."""
    # In the neuron can't fire, we're in the shadow realm!
    Et = 1000  # 1000 volts is infinity, for neurons.
    _, _, budget = adex(N, time, ns, ts, budget=True, Et=Et, **adex_kwargs)

    return budget


# TODO: add sigma
def adex(N,
         time,
         ns,
         ts,
         w_max=0.8e-9,
         tau_in=5e-3,
         bias=0.0e-9,
         Et=-50.0e-3,
         f=0,
         A=.1e-9,
         phi=0,
         sigma=0,
         C=200e-12,
         g_l=10e-9,
         V_l=-70e-3,
         V_reset=65e-3,
         V_max=-50e-3,
         a=0e-9,
         b=10e-12,
         tau_w=30e-3,
         E_rheo=-48e-3,
         delta_t=2e-3,
         time_step=1e-5,
         budget=True,
         report='text',
         save_args=None,
         step_params=None,
         seed=None):
    """A AdEx neuron
    
    Params
    ------
    time : Numeric
        Simulation run time (seconds)

    [...]

    step_params : None or 3-tuple (I, f, duty)
        Inject a set of square wave currect
    seed : None, int
        The random seed
    """
    np.random.seed(seed)

    defaultclock.dt = time_step * second
    prefs.codegen.target = 'numpy'

    # -----------------------------------------------------------------
    if save_args is not None:
        skip = ['ns', 'ts', 'save_args']
        arg_names = inspect.getargspec(adex)[0]

        args = []
        for arg in arg_names:
            if arg not in skip:
                row = (arg, eval(arg))
                args.append(row)

        with open("{}.csv".format(save_args), "w") as fi:
            writer = csv.writer(fi, delimiter=",")
            writer.writerows(args)

    # -----------------------------------------------------------------
    # If there's no input, return empty
    if ns.shape[0] == 0:
        return np.array([]), np.array([]), dict()

    # -----------------------------------------------------------------
    # tau_m:
    g_l *= siemens
    C *= farad
    tau_m = C / g_l

    # Other neuron params
    El = V_l * volt
    E_reset = V_reset * volt
    Et *= volt
    E_rheo *= volt

    # Comp vars
    w_max *= siemens
    tau_in *= second
    bias *= amp
    sigma *= siemens
    a *= siemens
    b *= amp
    delta_t *= volt
    tau_w *= second
    E_cut = Et + 8 * delta_t  # how high should the spike go?

    # osc injection?
    f *= Hz
    A *= amp
    phi *= second

    # -----------------------------------------------------------------
    # Define neuron and its connections
    eqs = """
    dv/dt = (g_l * (El - v) + g_l * delta_t * exp((v - Et) / delta_t) + I_in + I_osc + I_noise + I_ext + bias - w) / C : volt
    dw/dt = (a * (v - El) - w) / tau_w : amp
    I_in = g_in * (v - El) : amp
    I_noise = g_noise * (v - El) : amp
    dg_in/dt = -g_in / tau_in : siemens
    I_osc = A/2 * (1 + sin((t + phi) * f * 2 * pi)) : amp
    dg_noise/dt = -(g_noise + (sigma * sqrt(tau_in) * xi)) / tau_in : siemens
    """

    # Step injection?
    if step_params is not None:
        I, f_wave, duty = step_params
        waves = step_waves(I, f_wave, duty, time, time_step)
        I_sq = TimedArray(waves, dt=time_step * second)
        eqs += """I_ext = I_sq(t) * amp : amp"""
    else:
        eqs += """I_ext = 0 * amp : amp"""

    P_e = NeuronGroup(
        N,
        model=eqs,
        # refractory=2 * msecond,
        threshold='v > Et',
        reset="v = E_rheo; w += b",
        method='euler')

    # Init
    P_e.v = El
    P_e.w = a * (P_e.v - El)

    # Set up the 'network'
    P_stim = SpikeGeneratorGroup(np.max(ns) + 1, ns, ts * second)
    C_stim = Synapses(P_stim, P_e, on_pre='g_in += (w_max * rand())')
    C_stim.connect()

    # -----------------------------------------------------------------
    # Define variables
    spikes_e = SpikeMonitor(P_e)
    record = ['v', 'I_ext']
    traces_e = StateMonitor(P_e, record, record=True)

    # -----------------------------------------------------------------
    # Define basic net
    net = Network(P_e, traces_e)
    net.store('no_stim')

    # If budgets are desired, run the net without
    # any stimulation. (This strictly speaking isn't
    # necessary, but I can't get Brian to express the needed
    # diff eq to get the osc budget term in one pass.)
    if budget:
        net.run(time * second, report=report)
        V_osc = deepcopy(np.asarray(traces_e.v_))

    net.restore('no_stim')
    net.add([P_stim, C_stim, spikes_e])
    net.run(time * second, report=report)

    # -----------------------------------------------------------------
    # Analyze and save.

    # Extract spikes
    ns_e = np.asarray(spikes_e.i_)
    ts_e = np.asarray(spikes_e.t_)
    result = [ns_e, ts_e]

    if budget:
        # Define the terms that go into
        # the budget....
        E_leak = float(El)
        E_cut = float(E_cut)
        E_rheo = float(E_rheo)
        E_t = float(Et)

        V_max = float(V_max)
        V_m = np.asarray(traces_e.v_)

        # Rectify Vm
        V_m_thresh = V_m.copy()
        V_m_thresh[V_m_thresh > V_max] = V_max

        # Rectify V_osc
        V_osc[V_osc > V_max] = V_max

        # Est. Comp; 0 rectify
        V_comp = V_osc - V_m_thresh  # swtiched
        V_comp[V_comp > 0] = 0

        # Recenter osc so unit scale matches comp
        V_osc = E_leak - V_osc

        # Est free.
        V_free = V_max - V_m_thresh

        # Build vs
        vs = dict(
            tau_m=float(C / g_l),
            times=np.asarray(traces_e.t_),
            I_ext=np.asarray(traces_e.I_ext_),
            V_m=V_m,
            V_m_thresh=V_m_thresh,
            V_comp=V_comp,
            V_osc=V_osc,
            V_free=V_free,
            V_max=V_max,
            E_reset=E_reset,
            E_rheo=E_rheo,
            E_leak=E_leak,
            E_cut=E_cut,
            E_thresh=E_t)

        # Add the budget to the return var, result
        result.append(vs)

    return result
