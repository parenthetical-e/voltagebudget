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
# from voltagebudget.budget import filter_voltages
from voltagebudget.budget import locate_firsts
from voltagebudget.budget import budget_window
from voltagebudget.budget import locate_peaks
from voltagebudget.budget import estimate_communication
from voltagebudget.budget import precision
from voltagebudget.exp.autotune import autotune_V_osc


def forward_shadow(name,
                   N,
                   stim,
                   E,
                   t=0.4,
                   d=-5e-3,
                   w=2e-3,
                   T=0.0625,
                   f0=8,
                   A0=.05e-9,
                   phi0=np.pi,
                   mode='regular',
                   noise=False,
                   save_only=False,
                   verbose=False,
                   seed_value=42):
    """Optimize using the shadow voltage budget."""
    np.random.seed(seed_value)

    # --------------------------------------------------------------
    # Temporal params
    time_step = 1e-5
    coincidence_t = 1e-3

    # --------------------------------------------------------------
    if verbose:
        print(">>> Setting mode.")

    params, w_max, bias, sigma = read_modes(mode)
    if not noise:
        sigma = 0

    # --------------------------------------------------------------
    if verbose:
        print(">>> Importing stimulus.")

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
        opt_f=False,
        sigma=sigma,
        seed_value=seed_value,
        budget=True,
        save_args="{}_ref_args".format(name),
        time_step=time_step,
        **params)

    if ns_ref.size == 0:
        raise ValueError("The reference model didn't spike.")

    # -
    if verbose:
        print(">>> Creating shadow reference.")

    shadow_ref = shadow_adex(
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
        seed_value=seed_value,
        budget=True,
        save_args=False,
        time_step=time_step,
        **params)

    # --------------------------------------------------------------
    if verbose:
        print(">>> Begining budget estimates:")
    solutions = autotune_V_osc(
        N,
        t,
        E,
        d,
        w,
        stim,
        A0=A0,
        phi0=phi0,
        f0=f0,
        opt_f=opt_f,
        verbose=verbose)

    # --------------------------------------------------------------
    if verbose:
        print(">>> Analyzing results.")

    communication_scores = []
    computation_scores = []
    communication_voltages = []
    computation_voltages = []
    for n, sol in enumerate(solutions):
        # Unpack
        if opt_f:
            A_opt, phi_opt, f_opt = sol.x
        else:
            A_opt, phi_opt = sol.x
            f_opt = f0

        # Run 
        if verbose:
            print(">>> Running analysis for neuron {}/{}.".format(n + 1, N))

        # !
        ns_n, ts_n, budget_n = adex(
            N,
            t,
            ns,
            ts,
            w_max=w_m,
            bias=bias,
            f=f,
            A=A_m,
            phi=phi_m,
            sigma=sigma,
            budget=True,
            seed=seed_prob,
            report=report,
            time_step=time_step,
            save_args="{}_n_{}_opt_args".format(name, n),
            **params)

        # Analyze spikes
        comm = estimate_communication(
            ns_n,
            ts_n, (E + d, T),
            coincidence_t=coincidence_t,
            time_step=time_step)

        _, prec = precision(ns_m, ts_m, ns_ref, ts_ref, combine=True)

        # Extract budgets values
        voltages_m = budget_window(budget_m, E + d, w, select=None)
        V_comp = budget_reduce_fn(voltages_m['V_comp'])
        V_osc = budget_reduce_fn(voltages_m['V_osc'])

        # Store all stats for n
        communication_scores.append(comm)
        computation_scores.append(np.mean(prec))

        communication_voltages.append(V_osc)
        computation_voltages.append(V_comp)

    # --------------------------------------------------------------
    if verbose:
        print(">>> Saving results.")

    # Build a dict of results,
    results = {}
    results["communication_scores"] = communication_scores
    results["computation_scores"] = computation_scores
    results["communication_voltages"] = communication_voltages
    results["computation_voltages"] = computation_voltages

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