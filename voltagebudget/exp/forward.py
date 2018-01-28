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
from voltagebudget.util import mae
from voltagebudget.util import mad
from voltagebudget.exp.autotune import autotune_V_osc


def forward(name,
            stim,
            E_0,
            N=10,
            t=0.4,
            d=-5e-3,
            w=2e-3,
            T=0.0625,
            f=8,
            A_0=.05e-9,
            A_max=0.5e-9,
            phi_0=0,
            all_phi=False,
            mode='regular',
            noise=False,
            shadow=False,
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

    # If in shadow mode, replace ref voltages
    if shadow:
        voltages_ref = shadow_adex(
            N,
            t,
            ns,
            ts,
            w_in=w_in,
            bias_in=bias_in,
            f=0,
            A=0,
            phi=0,
            opt_phi=opt_phi,
            sigma=sigma,
            seed_value=seed_value,
            save_args=None,
            time_step=time_step,
            **params)

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

    # Filter ref spikes into the window of interest
    ns_ref, ts_ref = filter_spikes(ns_ref, ts_ref, (E, E + T))
    write_spikes("{}_ref_spks".format(name), ns_ref, ts_ref)

    if verbose:
        print(">>> {} spikes in the analysis window.".format(ns_ref.size))

    # --------------------------------------------------------------
    if verbose:
        print(">>> Begining budget estimates:")

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
        shadow=shadow,
        noise=noise,
        all_phi=all_phi,
        seed_value=seed_value,
        verbose=verbose)

    # --------------------------------------------------------------
    if verbose:
        print(">>> Analyzing results.")

    variances = []
    errors = []
    V_oscs = []
    V_comps = []
    V_frees = []
    As = []
    phis = []
    for n, sol in enumerate(solutions):
        A_opt, phi_opt, _ = sol

        # Run 
        if verbose:
            print(">>> Running analysis for neuron {}/{}.".format(n + 1, N))

        ns_n, ts_n, voltage_n = adex(
            N,
            t,
            ns,
            ts,
            w_in=w_in,
            bias_in=bias_in,
            f=f,
            A=A_opt,
            phi=phi_opt,
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

        write_spikes("{}_n_{}_spks".format(name, n), ns_n, ts_n)

        # Variance
        var = mad(ts_n)

        # Error
        error = mae(ts_n, ts_ref)

        # Extract budget values
        budget_n = budget_window(voltage_n, E + d, w, select=None)
        V_osc = np.abs(np.mean(budget_n['V_osc'][n, :]))
        V_comp = np.abs(np.mean(budget_n['V_comp'][n, :]))
        V_free = np.abs(np.mean(budget_n['V_free'][n, :]))

        # Store all stats for n
        variances.append(var)
        errors.append(np.mean(error))

        V_oscs.append(V_osc)
        V_comps.append(V_comp)
        V_frees.append(V_free)

        As.append(A_opt)
        phis.append(phi_opt)

        if verbose:
            print(
                ">>> (A {:0.12f}, phi {:0.3f})  ->  (N spks, {}, mae {:0.5f}, mad, {:0.5f})".
                format(A_opt, phi_opt, ns_n.size, error, var))

    # --------------------------------------------------------------
    if verbose:
        print(">>> Saving results.")

    # Build a dict of results,
    results = {}
    results["N"] = list(range(N))
    results["variances"] = variances
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