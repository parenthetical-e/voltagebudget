import os
import json
import csv
import numpy as np

import voltagebudget

from scipy.optimize import least_squares
from pyswarm import pso

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
from voltagebudget.util import select_n
from voltagebudget.util import mae
from voltagebudget.util import mad
from voltagebudget.util import score_by_group
from voltagebudget.util import score_by_n
from voltagebudget.exp.autotune import autotune_V_osc
from voltagebudget.plc import max_deviant


def reverse(name,
            stim,
            E_0,
            percent_change=0.1,
            N=10,
            t=0.4,
            d=-2e-3,
            w=2e-3,
            T=0.0625,
            f_0=8,
            f_max=60,
            A_0=.05e-9,
            A_max=.2e-9,
            scale=5,
            opt_phi=False,
            opt_f=False,
            opt_bias=False,
            mode='regular',
            noise=False,
            save_only=False,
            save_spikes=False,
            score_group=False,
            verbose=False,
            seed_value=42):

    np.random.seed(seed_value)

    if T < (1.0 / f_0):
        raise ValueError("T must be >= 1/f (osc. period).")

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

    ns_ref, ts_ref = adex(
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
        budget=False,
        save_args="{}_ref_args".format(name),
        time_step=time_step,
        seed_value=seed_value,
        **params)

    if ns_ref.size == 0:
        raise ValueError("The reference model didn't spike.")

    # -----------------------------------------------
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

    # -----------------------------------------------
    # Get opt ts_opt
    if verbose:
        print(">>> Finding optimal coordination.")
    initial, target, obs, ts_opt = max_deviant(ts_ref, percent_change)

    # -----------------------------------------------
    if verbose:
        print(">>> Finding phi_E.")
    phi_E = float(-E * 2 * np.pi * f_0)

    # -----------------------------------------------
    if verbose:
        print(">>> Starting the optimization.")

    # -
    if verbose:
        print(">>> Building initial values.")

    # set p_0
    p_0 = (A_0, phi_E, f_0, bias_in)

    # set bounds
    bounds = (np.zeros_like(p_0), np.zeros_like(p_0))
    bounds[0][0] = 0.0
    bounds[1][0] = A_max

    if opt_phi:
        bounds[0][1] = phi_E - np.pi / 2
        bounds[1][1] = phi_E + np.pi / 2
    else:
        # fix, essentially
        bounds[0][1] = phi_E - 1e-3
        bounds[1][1] = phi_E + 1e-3

    if opt_f:
        bounds[0][2] = 1
        bounds[1][2] = f_max
    else:
        # fix, essentially
        bounds[0][2] = f_0 - 1e-6
        bounds[1][2] = f_0 + 1e-6

    if opt_bias:
        if bias_in > 0:
            # Doubling the default gives a large range
            bounds[0][3] = -2 * bias_in
            bounds[1][3] = 2 * bias_in
        elif bias_in < 0:
            bounds[0][3] = 2 * bias_in
            bounds[1][3] = -2 * bias_in
        else:
            bounds[0][3] = -1e-14
            bounds[1][3] = 1e-14
    else:
        # fix, essentially
        if bias_in > 0:
            bounds[0][3] = bias_in - 1e-14
            bounds[1][3] = bias_in + 1e-14
        elif bias_in < 0:
            bounds[0][3] = bias_in + 1e-14
            bounds[1][3] = bias_in - 1e-14
        else:
            bounds[0][3] = -1e-14
            bounds[1][3] = 1e-14

    if verbose:
        print(">>> p_0: {}".format(p_0))
        print(">>> min: {}".format(bounds[0]))
        print(">>> max: {}".format(bounds[1]))

    solutions = []
    for n in range(N):
        # nth target
        _, ts_opt_n = select_n(n, ns_ref, ts_opt)

        if verbose:
            print(">>> Building problem ({}/{}).".format(n + 1, N))

        def problem(p):
            # Process
            A = p[0]
            phi = p[1]
            f = p[2]
            bias = p[3]

            # and run p.
            ns_p, ts_p = adex(
                N,
                t,
                ns,
                ts,
                A=A,
                phi=phi,
                f=f,
                w_in=w_in,
                bias_in=bias,
                sigma=sigma,
                seed_value=seed_value,
                budget=False,
                **params)

            # Err!
            _, ts_p_n = select_n(n, ns_p, ts_p)
            # err = mae(ts_p_n, ts_opt_n)

            min_l = min(len(ts_opt_n), len(ts_p_n))
            err = (ts_opt_n[:min_l] - ts_p_n[:min_l])

            if verbose:
                print(
                    ">>> (A {:0.15f}, phi {:.4f}, f {:.2f}, bias {:0.14f})  ->  (loss {:6})".
                    format(A, phi, f, bias, np.mean(err)))

            err = np.sqrt(np.sum(err**2))

            return err

        # -
        # -
        if verbose:
            print(">>> Finding solution.")

        # sol = least_squares(problem, p_0, bounds=bounds)
        # A_opt_n, phi_opt_n, f_opt_n, bias_opt_n = sol.x

        xopt, fopt = pso(problem, bounds[0], bounds[1])
        A_opt_n, phi_opt_n, f_opt_n, bias_opt_n = xopt

        solutions.append((A_opt_n, phi_opt_n, f_opt_n, bias_opt_n))

    # -----------------------------------------------
    if verbose:
        print(">>> Analyzing results.")

    variances = []
    errors = []
    n_spikes = []
    V_oscs = []
    V_comps = []
    V_frees = []
    As = []
    phis = []
    phis_w = []
    for n, sol in enumerate(solutions):
        A_opt_n, phi_opt_n, f_opt_n, bias_opt_n = sol

        # Run 
        if verbose:
            print(">>> Running analysis for neuron {}/{}.".format(n + 1, N))

        ns_n, ts_n, voltage_n = adex(
            N,
            t,
            ns,
            ts,
            w_in=w_in,
            bias_in=bias_opt_n,
            f=f_opt_n,
            A=A_opt_n,
            phi=phi_opt_n,
            sigma=sigma,
            budget=True,
            seed_value=seed_value,
            time_step=time_step,
            save_args="{}_n_{}_opt_args".format(name, n),
            **params)

        # Analyze spikes
        # Filter spikes in E    
        ns_ref, ts_opt = filter_spikes(ns_ref, ts_opt, (E, E + T))
        ns_n, ts_n = filter_spikes(ns_n, ts_n, (E, E + T))

        # Select n
        ns_ref_n, ts_opt_n = select_n(n, ns_ref, ts_opt)
        ns_n, ts_n = select_n(n, ns_n, ts_n)

        var, error = score_by_group(ts_opt_n, ts_n)

        if save_spikes:
            write_spikes("{}_n_{}_spks".format(name, n), ns_n, ts_n)

        # Extract budget values
        budget_n = budget_window(voltage_n, E + d, w, select=None)
        V_osc = np.abs(np.mean(budget_n['V_osc'][n, :]))
        V_comp = np.abs(np.mean(budget_n['V_comp'][n, :]))
        V_free = np.abs(np.mean(budget_n['V_free'][n, :]))

        # Store all stats for n
        variances.append(var)
        errors.append(np.mean(error))
        n_spikes.append(ts_n.size)

        V_oscs.append(V_osc)
        V_comps.append(V_comp)
        V_frees.append(V_free)

        As.append(A_opt_n)
        phis.append(phi_opt_n)

        if verbose:
            print(
                ">>> (A {:0.12f}, phi {:0.3f}, f{:1f}, b {:14f})  ->  (N spks, {}, mae {:0.5f}, mad, {:0.5f})".
                format(A_opt_n, phi_opt_n, f_opt_n, bias_opt_n, ns_n.size,
                       error, var))

    # --------------------------------------------------------------
    if verbose:
        print(">>> Saving results.")

    # Build a dict of results,
    results = {}
    results["N"] = list(range(N))
    results["variances"] = variances
    results["errors"] = errors
    results["n_spikes"] = n_spikes
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
        # Run

        # Analyze
