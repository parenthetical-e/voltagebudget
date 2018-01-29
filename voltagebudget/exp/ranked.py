import os
import json
import csv
import numpy as np

import voltagebudget
from voltagebudget.neurons import adex
from voltagebudget.neurons import shadow_adex
from voltagebudget.util import poisson_impulse
from voltagebudget.util import read_results
from voltagebudget.util import read_stim
from voltagebudget.util import read_args
from voltagebudget.util import read_modes
from voltagebudget.util import nearest_spike
from voltagebudget.util import write_spikes

from voltagebudget.util import locate_firsts
from voltagebudget.util import filter_spikes
from voltagebudget.util import budget_window
from voltagebudget.util import locate_peaks
from voltagebudget.util import estimate_communication
from voltagebudget.util import precision
from voltagebudget.util import mad
from voltagebudget.util import mae
from voltagebudget.util import select_n
from voltagebudget.exp.autotune import autotune_V_osc

from scipy.optimize import least_squares


def ranked(name,
           stim,
           E_0,
           rank=0,
           t=0.4,
           d=-5e-3,
           w=2e-3,
           T=0.0625,
           f=8,
           A_0=.05e-9,
           A_max=0.5e-9,
           phi_0=1.57,
           N=10,
           opt_phi=False,
           mode='regular',
           noise=False,
           save_only=False,
           verbose=False,
           seed_value=42):
    """Optimize using the shadow voltage budget.

    TODO: add shadow mode?
    """
    np.random.seed(seed_value)

    # --------------------------------------------------------------
    # Temporal params
    time_step = 1e-5
    coincidence_t = 1e-3

    # Process rank
    if verbose:
        print(">>> Target rank is {}".format(rank))

    rank -= 1  # Adj for python zero indexing

    if rank > N:
        raise ValueError("rank must be less than N")
    if rank < 0:
        raise ValueError("rank must be > 0.")

    # --------------------------------------------------------------
    if verbose:
        print(">>> Setting mode.")

    params, w_in, bias_in, sigma = read_modes(mode)
    if not noise:
        sigma = 0

    # --------------------------------------------------------------
    if verbose:
        print(">>> Importing stimulus from {}.".format(stim))

    stim_data = read_stim(stim)
    ns = np.asarray(stim_data['ns'])
    ts = np.asarray(stim_data['ts'])

    # --------------------------------------------------------------
    # Define target computation (i.e., no oscillation)
    # (TODO Make sure and explain this breakdown well in th paper)
    if verbose:
        print(">>> Creating reference spikes.")

    ns_ref, ts_ref, voltages_ref = adex(
        N,
        t,
        ns,
        ts,
        w_in=w_in,
        bias_in=bias_in,
        f=0,
        A=0,
        phi=0,
        sigma=sigma,
        budget=True,
        save_args="{}_ref_args".format(name),
        time_step=time_step,
        seed_value=seed_value,
        **params)

    if ns_ref.size == 0:
        raise ValueError("The reference model didn't spike.")

    # Find the ref spike closest to E_0
    # and set that as E
    if np.isclose(E_0, 0.0):
        _, E = locate_firsts(ns_ref, ts_ref, combine=True)
        if verbose:
            print(">>> Locking on first spike. E was {}.".format(E))
    else:
        E = nearest_spike(ts_ref, E_0)
        if verbose:
            print(">>> E_0 was {}, using closest at {}.".format(E_0, E))

    # Find the phase begin a osc cycle at E 
    phi_E = float(-E * 2 * np.pi * f)

    # Filter ref spikes into the window of interest
    ns_ref, ts_ref = filter_spikes(ns_ref, ts_ref, (E, E + T))
    write_spikes("{}_ref_spks".format(name), ns_ref, ts_ref)

    if verbose:
        print(">>> {} spikes in the analysis window.".format(ns_ref.size))

    # --------------------------------------------------------------
    # Rank the neurons
    budget_ref = budget_window(voltages_ref, E + d, w, select=None)

    # Pick the neuron to optimize, based on its rank
    idx = np.argsort([budget_ref["V_free"][j, :].mean() for j in range(N)])
    n = int(np.where(idx == rank)[0])
    if verbose:
        print(">>> The Vf rank index: {}.".format(idx))
        print(">>> Rank {} is neuron {}.".format(rank + 1, n))

        solutions = autotune_V_osc(
            N,
            t,
            E,
            d,
            ns,
            ts,
            voltages_ref,
            A_0=A_0,
            A_max=A_max,
            phi_0=phi_0,
            f=f,
            select_n=n,
            noise=noise,
            seed_value=seed_value,
            verbose=verbose)

    A_hat, phi_hat, _ = solutions[0]

    # --------------------------------------------------------------
    if verbose:
        print(">>> Analyzing results.")

    ns_n, ts_n, voltage_n = adex(
        N,
        t,
        ns,
        ts,
        w_in=w_in,
        bias_in=bias_in,
        f=f,
        A=A_hat,
        phi=phi_E,
        sigma=sigma,
        budget=True,
        seed_value=seed_value,
        time_step=time_step,
        save_args="{}_n_{}_opt_args".format(name, n),
        **params)

    # Analyze spikes
    # Filter spikes in E
    ns_ref, ts_ref = filter_spikes(ns_ref, ts_ref, (E, E + T))
    ns_n, ts_n = filter_spikes(ns_n, ts_n, (E, E + T))
    write_spikes("{}_rank_{}_spks".format(name, n), ns_n, ts_n)

    variances = []
    delta_variances = []
    errors = []
    V_oscs = []
    V_comps = []
    V_frees = []
    As = []
    phis = []
    phis_w = []
    for i in range(N):
        ns_ref_i, ts_ref_i = select_n(i, ns_ref, ts_ref)
        ns_i, ts_i = select_n(i, ns_n, ts_n)

        # Variance
        var = mad(ts_i)
        var_ref = mad(ns_ref_i)
        delta_var = var_ref - var

        # Error
        error = mae(ts_i, ts_ref_i)

        # Extract budget values
        budget_n = budget_window(voltage_n, E + d, w, select=None)
        V_osc = np.abs(np.mean(budget_n['V_osc'][i, :]))
        V_comp = np.abs(np.mean(budget_n['V_comp'][i, :]))
        V_free = np.abs(np.mean(budget_n['V_free'][i, :]))

        # Store all stats for n
        variances.append(var)
        delta_variances.append(delta_var)
        errors.append(np.mean(error))

        V_oscs.append(V_osc)
        V_comps.append(V_comp)
        V_frees.append(V_free)

        As.append(A_hat)

        # Below are constant in this exp, including for consistency with other
        # exps (pareto, forward, etc)
        phis.append(phi_E)
        phis_w.append(phi_hat)

        if verbose:
            print(
                u">>> (n {})  ->  (N spks, {}, mae {:0.5f}, delta_mad, {:0.5f})".
                format(i, ns_n.size, error, delta_var))

    # --------------------------------------------------------------
    if verbose:
        print(">>> Saving results.")

    # Build a dict of results,
    results = {}
    results["N"] = list(range(N))
    results["variances"] = variances
    results["delta_variances"] = delta_variances
    results["errors"] = errors

    results["V_osc"] = V_oscs
    results["V_comp"] = V_comps
    results["V_free"] = V_frees

    results["As"] = As
    results["phis"] = phis

    # then write it out.
    keys = sorted(results.keys())
    with open("{}.csv".format(name), "w") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(keys)
        writer.writerows(zip(* [results[key] for key in keys]))

    # If running in a CL, returns are line noise?
    if not save_only:
        return results
    else:
        return None