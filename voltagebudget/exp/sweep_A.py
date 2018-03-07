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
from voltagebudget.util import write_spikes
from voltagebudget.util import write_voltages
from voltagebudget.util import mad
from voltagebudget.util import mae
from voltagebudget.util import select_n
from voltagebudget.util import score_by_group
from voltagebudget.util import score_by_n
from voltagebudget.util import find_E
from voltagebudget.util import find_phis

from voltagebudget.exp.autotune import autotune_V_osc

from scipy.optimize import least_squares


def sweep_A(name,
            stim,
            E_0,
            A_0=0.00e-9,
            A_max=0.5e-9,
            n_samples=10,
            t=0.4,
            d=-5e-3,
            w=2e-3,
            T=0.0625,
            f=8,
            N=10,
            mode='regular',
            noise=False,
            no_lock=False,
            verbose=False,
            save_only=False,
            save_details=False,
            seed_value=42):
    """Optimize using the shadow voltage budget."""

    np.random.seed(seed_value)

    # --------------------------------------------------------------
    # Temporal params
    time_step = 1e-5

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
        save_args=None,
        time_step=time_step,
        seed_value=seed_value,
        **params)

    if ns_ref.size == 0:
        raise ValueError("The reference model didn't spike.")

    # --------------------------------------------------------------
    # Find E and phis
    E = find_E(E_0, ns_ref, ts_ref, no_lock=no_lock, verbose=verbose)
    phi_w, phi_E = find_phis(E, f, d, verbose=verbose)

    # Filter ref spikes into the window of interest
    ns_ref, ts_ref = filter_spikes(ns_ref, ts_ref, (E, E + T))
    if verbose:
        print(">>> {} spikes in the analysis window.".format(ns_ref.size))

    # --------------------------------------------------------------
    # Rank the neurons
    budget_ref = budget_window(voltages_ref, E + d, w, select=None)

    # --------------------------------------------------------------
    samples = np.linspace(A_0, A_max, n_samples)

    variances = []
    errors = []
    n_spikes = []
    V_oscs = []
    V_osc_refs = []
    V_comps = []
    V_comp_refs = []
    V_frees = []
    V_budgets = []
    As = []
    biases = []
    phis = []
    phis_w = []
    for i, A_i in enumerate(samples):
        # Run 
        if verbose:
            print(">>> Running A {:0.15f} ({}/{}).".format(A_i, i + 1,
                                                           n_samples))
        # Spikes, using phi_E
        ns_i, ts_i = adex(
            N,
            t,
            ns,
            ts,
            w_in=w_in,
            bias_in=bias_in,
            f=f,
            A=A_i,
            phi=phi_E,
            sigma=sigma,
            budget=False,
            seed_value=seed_value,
            time_step=time_step,
            save_args=None,
            **params)

        # Voltages at E+d, using phi_w
        _, _, voltage_i = adex(
            N,
            t,
            ns,
            ts,
            w_in=w_in,
            bias_in=bias_in,
            f=f,
            A=A_i,
            phi=phi_w,
            sigma=sigma,
            budget=True,
            seed_value=seed_value,
            time_step=time_step,
            save_args=None,
            **params)

        # -------------------------------------------------------------------
        # Analyze!

        # Filter spikes in E    
        ns_i, ts_i = filter_spikes(ns_i, ts_i, (E, E + T))

        # Want group var(ts_i)
        var = mad(ts_i)

        # But avg of individual {n in N} errors
        _, error = score_by_n(N, ns_ref, ts_ref, ns_i, ts_i)

        # Save scores
        variances.append(var)
        errors.append(np.mean(error))
        n_spikes.append(ts_i.size)

        # -------------------------------------------------------------------
        # Extract budget values and save 'em
        # ith
        budget_i = budget_window(voltage_i, E + d, w, select=None)

        V_b = float(voltage_i['V_budget'])
        V_osc = np.abs(np.mean(budget_i['V_osc']))
        V_comp = np.abs(np.mean(budget_i['V_comp']))
        V_free = np.abs(np.mean(budget_i['V_free']))

        # ref
        V_comp_ref = np.abs(np.mean(budget_ref['V_comp']))
        V_osc_ref = np.abs(np.mean(budget_ref['V_osc']))

        # Save 'em all
        V_oscs.append(V_osc)
        V_osc_refs.append(V_osc_ref)

        V_comps.append(V_comp)
        V_comp_refs.append(V_comp_ref)

        V_frees.append(V_free)
        V_budgets.append(V_b)
        As.append(A_i)

        # These will repeat for each i, but that's ok.
        biases.append(bias_in)
        phis.append(phi_E)
        phis_w.append(phi_w)

        # -------------------------------------------------------------------
        if verbose:
            print(
                ">>> (A {:0.12f})  ->  (N spks, {}, mae {:0.5f}, mad, {:0.5f})".
                format(A_i, ns_i.size / float(N), error, var))

        if save_details:
            print(">>> Writing details for A {} (nA)".format(
                np.round(A_i * 1e9, 3)))

            write_spikes("{}_A{}_spks".format(name, np.round(A_i * 1e9, 3)),
                         ns_i, ts_i)

            write_voltages(
                "{}_A{}".format(name, np.round(A_i * 1e9, 3)),
                voltage_i,
                select=["V_comp", "V_osc", "V_m"])

    # --------------------------------------------------------------
    if verbose:
        print(">>> Saving results.")

    # Build a dict of results,
    results = {}
    results["variances"] = variances
    results["errors"] = errors
    results["n_spikes"] = n_spikes

    results["V_osc"] = V_oscs
    results["V_osc_ref"] = V_osc_refs

    results["V_comp"] = V_comps
    results["V_comp_ref"] = V_comp_refs

    results["V_free"] = V_frees
    results["V_b"] = V_budgets

    results["As"] = As
    results["biases"] = biases
    results["phis"] = phis
    results["phis_w"] = phis_w

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