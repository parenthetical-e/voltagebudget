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
from voltagebudget.util import score_by_group
from voltagebudget.util import score_by_n
from voltagebudget.exp.autotune import autotune_V_osc

from scipy.optimize import least_squares
from pyswarm import pso


def min_max(name,
            stim,
            E_0,
            A_0=0.00e-9,
            A_max=0.3e-9,
            t=0.4,
            d=-5e-3,
            w=2e-3,
            T=0.0625,
            f=8,
            percent=0.5,
            n_samples=10,
            N=10,
            mode='regular',
            target='min',
            noise=False,
            scale_w_in=1,
            correct_bias=False,
            save_only=False,
            verbose=False,
            seed_value=42):
    """Fit A to the min free voltage, then explore a 
    range of A around this point.
    """
    # --------------------------------------------------------------
    # Init
    # Seed
    np.random.seed(seed_value)

    # Temporal params
    time_step = 1e-5

    # --------------------------------------------------------------
    if verbose:
        print(">>> Setting mode.")

    params, w_in, bias_in, sigma = read_modes(mode)
    if not noise:
        sigma = 0

    # Scale input weight
    w_in = [scale_w_in * w_in[0], scale_w_in * w_in[1]]

    # --------------------------------------------------------------
    if verbose:
        print(">>> Importing stimulus from {}.".format(stim))

    stim_data = read_stim(stim)
    ns = np.asarray(stim_data['ns'])
    ts = np.asarray(stim_data['ts'])

    # --------------------------------------------------------------
    # Define target computation (i.e., no oscillation)
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
    # Find the ref spike closest to E_0
    # and set that as E
    if np.isclose(E_0, 0.0):
        _, E = locate_firsts(ns_ref, ts_ref, combine=True)
        if verbose:
            print(">>> Locking on first spike. E was {}.".format(E))
    else:
        E = nearest_spike(ts_ref, E_0)
        if verbose:
            print(
                ">>> ts_ref min {}, max {}".format(ts_ref.min(), ts_ref.max()))
            print(">>> E_0 was {}, using closest at {}.".format(E_0, E))

    # --------------------------------------------------------------
    # Find the phase begin a osc cycle at E 
    phi_E = float(-E * 2 * np.pi * f)
    phi_w = float((-(E + d) * 2 * np.pi * f) + np.pi / 2)

    # --------------------------------------------------------------
    # Filter ref spikes into the window of interest
    ns_ref, ts_ref = filter_spikes(ns_ref, ts_ref, (E, E + T))
    if verbose:
        print(">>> {} spikes in the analysis window.".format(ns_ref.size))

    # --------------------------------------------------------------
    # Rank the neurons
    budget_ref = budget_window(voltages_ref, E + d, w, select=None)

    # Pick the neuron/V_free to optimize, based on its rank
    if verbose:
        print(">>> Optimization target is '{} V_free'".format(target))

    if target == 'min':
        n = np.argmin([budget_ref["V_free"][j, :].mean() for j in range(N)])
    elif target == 'max':
        n = np.argmax([budget_ref["V_free"][j, :].mean() for j in range(N)])
    else:
        raise ValueError("target must be min or max")

    # --------------------------------------------------------------
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
        phi_0=phi_w,
        f=f,
        select_n=n,
        correct_bias=correct_bias,
        noise=noise,
        seed_value=seed_value,
        verbose=verbose)

    A_hat, _ = solutions[0]
    if verbose:
        print(">>> A_hat: {:0.15f}".format(A_hat))

    if correct_bias:
        bias = bias_in - (A_hat / 2.0)
        if verbose:
            print(">>> (bias {}) -> (bias_adj {})".format(bias_in, bias))
    else:
        bias = bias_in

    # --------------------------------------------------------------
    # Sample around A_hat
    A_low = A_hat - (A_hat * percent)
    A_high = A_hat + (A_hat * percent)
    samples = np.linspace(A_low, A_high, n_samples)

    samples[samples > A_max] = A_max
    samples[samples < A_0] = A_0

    # -
    variances = []
    errors = []
    n_spikes = []
    V_oscs = []
    V_comps = []
    V_frees = []
    V_budgets = []
    As = []
    biases = []
    phis = []
    phis_w = []
    for i, A_i in enumerate(samples):
        if verbose:
            print(">>> Running A {:0.18f} ({}/{}).".format(A_i, i + 1,
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

        # Voltages at E + d, using phi_w
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
        ns_ref, ts_ref = filter_spikes(ns_ref, ts_ref, (E, E + T))
        ns_i, ts_i = filter_spikes(ns_i, ts_i, (E, E + T))

        # Want group var(ts_i)
        var, _ = score_by_group(ts_ref, ts_i)

        # But avg of individual {n in N} errors
        _, error = score_by_n(N, ns_ref, ts_ref, ns_i, ts_i)

        # Save scores
        variances.append(var)
        errors.append(np.mean(error))
        n_spikes.append(ts_i.size)

        # Extract budget values and save 'em
        budget_i = budget_window(voltage_i, E + d, w, select=None)
        V_osc = np.abs(np.mean(budget_i['V_osc'][i, :]))
        V_comp = np.abs(np.mean(budget_i['V_comp'][i, :]))
        V_free = np.abs(np.mean(budget_i['V_free'][i, :]))
        V_b = float(voltage_i['V_budget'])

        V_oscs.append(V_osc)
        V_comps.append(V_comp)
        V_frees.append(V_free)
        V_budgets.append(V_b)
        As.append(A_i)

        # These will repeat for each i, but that's ok.
        biases.append(bias_in)
        phis.append(phi_E)
        phis_w.append(phi_w)

        if verbose:
            print(
                ">>> (A {:0.18f})  ->  (N spks, {}, mae {:0.5f}, mad, {:0.5f})".
                format(A_i, ns_i.size / float(N), error, var))

    # --------------------------------------------------------------
    if verbose:
        print(">>> Saving results.")

    # Build a dict of results,
    results = {}
    results["variances"] = variances
    results["errors"] = errors
    results["n_spikes"] = n_spikes

    results["V_osc"] = V_oscs
    results["V_comp"] = V_comps
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