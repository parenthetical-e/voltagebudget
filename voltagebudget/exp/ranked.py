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
           phi_0=0,
           N=10,
           opt_phi=False,
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

    # Process rank
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
    # Rank the neurons
    budget_ref = budget_window(voltages_ref, E + d, w, select=None)

    # Pick the neuron to optimize, based on its rank
    n = np.argmin([budget_ref["V_free"][j, :].mean() for j in range(N)])
    if verbose:
        print(">>> Rank {} is neuron {}.".format(rank, n))

    # least_squares() was struggling with small A, so boost it
    # for param search purposes, then divide it back out in the 
    # problem definition
    rescale = 1e10

    # ---------------------------------------------------------------
    # Define a loss func (closing several locals)
    def est_loss(voltage):
        # Select window
        budget = budget_window(voltage, E + d, w, select=None)

        # Get budget terms for opt
        V_free = np.abs(budget_ref['V_free'][n, :])
        V_osc = np.abs(budget['V_osc'][n, :])

        # loss = np.mean(V_free - V_osc)
        loss = V_free - V_osc

        return loss

    def phi_problem(p, A):
        """A new problem for each neuron"""

        phi = p[0]
        _, _, voltage = adex(
            N,
            t,
            ns,
            ts,
            A=A / rescale,
            phi=phi,
            f=f,
            w_in=w_in,
            bias_in=bias_in,
            sigma=sigma,
            seed_value=seed_value,
            budget=True,
            **params)

        loss = est_loss(voltage)

        if verbose:
            print(">>> (A {:0.12f}, phi {:0.3f})  ->  (loss {})".format(
                A / rescale, phi, np.sum(loss)))

        return loss

    def A_problem(p, phi):
        """A new problem for each neuron"""

        A = p[0] / rescale

        _, _, voltage = adex(
            N,
            t,
            ns,
            ts,
            A=A,
            phi=phi,
            f=f,
            w_in=w_in,
            bias_in=bias_in,
            sigma=sigma,
            seed_value=seed_value,
            budget=True,
            **params)

        loss = est_loss(voltage)

        if verbose:
            print(">>> (A {:0.12f}, phi {:0.3f})  ->  (loss {})".format(
                A / rescale, phi, np.sum(loss)))

        return loss

    # Do the opt, phi then A for ith neuron
    # ---------------------------------------------------------------

    # Opt phi
    if verbose:
        print(">>> Optimizing phi.")

    p0 = [phi_0]
    bounds = (0, np.pi)
    sol = least_squares(
        lambda p: phi_problem(p, A_0 * rescale), p0, bounds=bounds)

    phi_hat = sol.x[0]

    # Opt A
    if verbose:
        print(">>> Optimizing A")

    p0 = [A_0 * rescale]
    bounds = (0, A_max * rescale)
    sol = least_squares(lambda p: A_problem(p, phi_hat), p0, bounds=bounds)

    A_hat = sol.x[0]

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
        A=A_hat / rescale,
        phi=phi_hat,
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
    errors = []
    V_oscs = []
    V_comps = []
    V_frees = []
    As = []
    phis = []
    for i in range(N):
        ns_ref_i, ts_ref_i = select_n(i, ns_ref, ts_ref)
        ns_i, ts_i = select_n(i, ns_n, ts_n)

        # Variance
        var = mad(ts_i)

        # Error
        error = mae(ts_i, ts_ref_i)

        # Extract budget values
        budget_n = budget_window(voltage_n, E + d, w, select=None)
        V_osc = np.abs(np.mean(budget_n['V_osc'][i, :]))
        V_comp = np.abs(np.mean(budget_n['V_comp'][i, :]))
        V_free = np.abs(np.mean(budget_n['V_free'][i, :]))

        # Store all stats for n
        variances.append(var)
        errors.append(np.mean(error))

        V_oscs.append(V_osc)
        V_comps.append(V_comp)
        V_frees.append(V_free)

        As.append(A_hat)
        phis.append(phi_hat)

        if verbose:
            print(">>> (n {})  ->  (N spks, {}, mae {:0.5f}, mad, {:0.5f})".
                  format(i, ns_n.size, error, var))

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