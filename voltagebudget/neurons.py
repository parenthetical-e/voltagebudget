import inspect
import csv
import numpy as np
from brian2 import *
from copy import deepcopy
from voltagebudget.util import step_waves


def shadow_adex(N, time, ns, ts, **adex_kwargs):
    """Est. the 'shadow voltage' of the AdEx membrane voltage."""
    # In the neuron can't fire, we're in the shadow realm!
    V_t = 1000  # 1000 volts is infinity, for neurons.
    _, _, budget = adex(N, time, ns, ts, budget=True, V_t=V_t, **adex_kwargs)

    return budget


def _parse_membrane_param(x, N, prng):
    try:
        if len(x) == 2:
            x_min, x_max = x
            x = prng.uniform(x_min, x_max, N)
        else:
            raise ValueError("Parameters must be scalars, or 2 V_lement lists")
    except TypeError:
        pass

    return x, prng


# TODO: add sigma
def adex(N,
         time,
         ns,
         ts,
         w_in=0.8e-9,
         tau_in=5e-3,
         bias_in=0.0e-9,
         V_t=-50.0e-3,
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
         V_rheo=-48e-3,
         delta_t=2e-3,
         time_step=1e-5,
         budget=True,
         report='text',
         save_args=None,
         step_params=None,
         seed_value=42):
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
    # -----------------------------------------------------------------
    # Plant all the seeds!
    seed(seed_value)
    prng = np.random.RandomState(seed_value)

    # Integration settings
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
            writer = csv.writer(fi, dV_limiter=",")
            writer.writerows(args)

    # -----------------------------------------------------------------
    # If there's no input, return empty
    if ns.shape[0] == 0:
        return np.array([]), np.array([]), dict()

    # -----------------------------------------------------------------
    # Adex dynamics params
    g_l, prng = _parse_membrane_param(g_l, N, prng)
    C, prng = _parse_membrane_param(C, N, prng)

    # Potentially random synaptic params
    # Note: w_in gets created after synaptic input is 
    # Defined.
    bias_in, prng = _parse_membrane_param(bias_in, N, prng)
    tau_in, prng = _parse_membrane_param(tau_in, N, prng)

    # Potentially random membrane params
    V_rheo, prng = _parse_membrane_param(V_rheo, N, prng)
    a, prng = _parse_membrane_param(a, N, prng)
    b, prng = _parse_membrane_param(b, N, prng)
    delta_t, prng = _parse_membrane_param(delta_t, N, prng)
    tau_w, prng = _parse_membrane_param(tau_w, N, prng)

    # Fixed membrane dynamics
    sigma *= siemens
    V_cut = V_t + 8 * np.mean(delta_t)

    # Oscillation params
    f *= Hz
    A *= amp
    phi *= second

    # -----------------------------------------------------------------
    # Define an adex neuron, and its connections
    eqs = """
    dv/dt = (g_l * (V_l - v) + g_l * delta_t * exp((v - V_t) / delta_t) + I_in + I_osc + I_noise + I_ext + bias_in - w) / C : volt
    dw/dt = (a * (v - V_l) - w) / tau_w : amp
    I_in = g_in * (v - V_l) : amp
    I_noise = g_noise * (v - V_l) : amp
    dg_in/dt = -g_in / tau_in : siemens
    I_osc = A/2 * (1 + sin((t + phi) * f * 2 * pi)) : amp
    dg_noise/dt = -(g_noise + (sigma * sqrt(tau_in) * xi)) / tau_in : siemens
    C : farad
    g_l : siemens 
    a : siemens
    b : amp
    delta_t : volt
    tau_w : second
    V_rheo : volt
    bias_in : amp
    tau_in : second
    """

    # Step current injection?
    if step_params is not None:
        I, f_wave, duty = step_params
        waves = step_waves(I, f_wave, duty, time, time_step)
        I_sq = TimedArray(waves, dt=time_step * second)
        eqs += """I_ext = I_sq(t) * amp : amp"""
    else:
        eqs += """I_ext = 0 * amp : amp"""

    P_n = NeuronGroup(
        N,
        model=eqs,
        threshold='v > V_t',
        reset="v = V_rheo; w += b",
        method='euler')

    # Init adex params
    # Fixed voltages neuron params
    V_l *= volt
    V_reset *= volt
    V_t *= volt
    V_cut *= volt

    P_n.a = a * siemens
    P_n.b = b * amp
    P_n.delta_t = delta_t * volt
    P_n.tau_w = tau_w * second
    P_n.V_rheo = V_rheo * volt
    P_n.C = C * farad
    P_n.g_l = g_l * siemens
    P_n.bias_in = bias_in * amp
    P_n.tau_in = tau_in * second

    # Init V0, w0
    P_n.v = V_l
    P_n.w = P_n.a * (P_n.v - V_l)

    # -----------------------------------------------------------------
    # Add synaptic input into the network.
    P_stim = SpikeGeneratorGroup(np.max(ns) + 1, ns, ts * second)
    C_stim = Synapses(
        P_stim, P_n, model='w_in : siemens', on_pre='g_in += w_in')
    C_stim.connect()

    # (Finally) Potentially random weights
    w_in, prng = _parse_membrane_param(w_in, len(C_stim), prng)
    C_stim.w_in = w_in * siemens

    # -----------------------------------------------------------------
    # Record input and voltage 
    spikes_n = SpikeMonitor(P_n)
    record = ['v', 'I_ext']
    traces_n = StateMonitor(P_n, record, record=True)

    # -----------------------------------------------------------------
    # Build the model!
    net = Network(P_n, traces_n)
    net.store('no_stim')

    # If budgets are desired, run the net without
    # any stimulation. (This strictly speaking isn't
    # necessary, but I can't get Brian to express the needed
    # diff eq to get the osc budget term in one pass.)
    if budget:
        net.run(time * second, report=report)
        V_osc = deepcopy(np.asarray(traces_n.v_))

    net.restore('no_stim')
    net.add([P_stim, C_stim, spikes_n])
    net.run(time * second, report=report)

    # -----------------------------------------------------------------
    # Extract data from the run model

    # Spikes
    ns_e = np.asarray(spikes_n.i_)
    ts_e = np.asarray(spikes_n.t_)

    # Define the return objects
    result = [ns_e, ts_e]

    # Define the terms that go into the budget
    # (these get added to result at the end)
    if budget:
        V_leak = float(V_l)
        V_cut = float(V_cut)
        V_rheo = float(V_rheo)
        V_t = float(V_t)

        V_max = float(V_max)
        V_m = np.asarray(traces_n.v_)

        # Rectify Vm
        V_m_thresh = V_m.copy()
        V_m_thresh[V_m_thresh > V_max] = V_max

        # Rectify V_osc
        V_osc[V_osc > V_max] = V_max

        # Est. Comp; 0 rectify
        V_comp = V_osc - V_m_thresh  # swtiched
        V_comp[V_comp > 0] = 0

        # Recenter osc so unit scale matches comp
        V_osc = V_leak - V_osc

        # Est free.
        V_free = V_max - V_m_thresh

        # Build budget dict
        vs = dict(
            tau_m=float(C / g_l),
            times=np.asarray(traces_n.t_),
            I_ext=np.asarray(traces_n.I_ext_),
            V_m=V_m,
            V_m_thresh=V_m_thresh,
            V_comp=V_comp,
            V_osc=V_osc,
            V_free=V_free,
            V_max=V_max,
            V_reset=V_reset,
            V_rheo=V_rheo,
            V_leak=V_leak,
            V_cut=V_cut,
            V_thresh=V_t)

        # Add the budget to the return var, result
        result.append(vs)

    return result
