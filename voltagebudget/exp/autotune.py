import json
import csv
import os
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

from voltagebudget.util import select_n
from voltagebudget.util import filter_voltages
from voltagebudget.util import filter_spikes
from voltagebudget.util import budget_window
from voltagebudget.util import locate_firsts
from voltagebudget.util import locate_peaks
from voltagebudget.util import find_E
from voltagebudget.util import find_phis


def autotune_homeostasis(stim,
                         target,
                         E_0=0,
                         N=250,
                         t=0.4,
                         A=0.05e-9,
                         Z_0=1e-6,
                         Z_max=1,
                         f=8,
                         n_jobs=1,
                         mode='regular',
                         noise=False,
                         no_lock=False,
                         verbose=False,
                         seed_value=42):
    """Find the optimal Z value for a given (A, f)."""

    np.random.seed(seed_value)

    # --------------------------------------------------------------
    # Temporal params
    time_step = 1e-5

    # ---------------------------------------------------------------
    if verbose:
        print(">>> Setting mode.")

    params, w_in, bias_in, sigma = read_modes(mode)
    if not noise:
        sigma = 0

    # ---------------------------------------------------------------
    if verbose:
        print(">>> Importing stimulus from {}.".format(stim))

    stim_data = read_stim(stim)
    ns = np.asarray(stim_data['ns'])
    ts = np.asarray(stim_data['ts'])

    # ---------------------------------------------------------------
    if verbose:
        print(">>> Creating reference spikes.")

    ns_ref, ts_ref = adex(
        N,
        t,
        ns,
        ts,
        w_in=w_in,
        bias_in=bias_in,
        f=0.0,
        A=0,
        phi=0,
        sigma=sigma,
        budget=False,
        save_args=None,
        time_step=time_step,
        seed_value=seed_value,
        **params)

    if ns_ref.size == 0:
        raise ValueError("The reference model didn't spike.")

    # --------------------------------------------------------------
    # Find T, E and phis
    T = 1 / float(f)  # Analysis window
    E = find_E(E_0, ns_ref, ts_ref, no_lock=no_lock, verbose=verbose)
    _, phi_E = find_phis(E, f, 0, verbose=verbose)

    # Filter ref spikes into the window of interest
    ns_ref, ts_ref = filter_spikes(ns_ref, ts_ref, (E, E + T))
    if verbose:
        print(">>> {} spikes in the analysis window.".format(ns_ref.size))

    # ---------------------------------------------------------------
    def Z_problem(p):
        Z = p[0]
        bias = bias_in - (Z * A)

        ns_y, ts_y = adex(
            N,
            t,
            ns,
            ts,
            E=E,
            n_cycles=2,
            w_in=w_in,
            bias_in=bias,
            f=f,
            A=A,
            phi=phi_E,
            sigma=sigma,
            budget=False,
            save_args=None,
            time_step=time_step,
            seed_value=seed_value,
            **params)
        ns_y, ts_y = filter_spikes(ns_y, ts_y, (E, E + T))

        delta = float(abs(ts_ref.size - ts_y.size)) / N
        loss = abs(target - delta)

        if verbose:
            print("(Z {:0.18f}, bias_adj/bias {:0.18f}) -> (loss {:0.6f})".
                  format(Z, bias / bias_in, loss))

        # return np.sqrt(np.sum(loss**2))
        return loss

    # Opt init
    p0 = [0.1]
    bounds = (Z_0, Z_max)

    if verbose:
        print(">>> p0 {}".format(p0))
        print(">>> bounds {}".format(bounds))
        print(">>> Running the optimization")

    # Pso:
    # Using PSO for this is overkill. scipy.least_squares is misbehaving -
    # it won't step at all. Much fiddling has had no effect. Overkill
    # is better than no progress; pso has a similar API, so here we are.
    #
    # Hyper-params taken from
    # http://hvass-labs.org/people/magnus/publications/pedersen10good-pso.pdf
    xopt, _ = pso(
        Z_problem,
        [bounds[0]],
        [bounds[1]],
        swarmsize=25,
        omega=0.39,
        phip=2.5,
        phig=1.33,
        minfunc=1e-3,  # ...no need to be that precise
        processes=n_jobs)
    Z_hat = xopt[0]

    return Z_hat


def autotune_V_osc(N,
                   t,
                   E,
                   d,
                   ns,
                   ts,
                   voltage_ref,
                   w=2e-3,
                   A_0=0.05e-9,
                   A_max=0.5e-9,
                   phi_0=1.57,
                   f=8,
                   mode='regular',
                   select_n=None,
                   noise=False,
                   correct_bias=False,
                   seed_value=42,
                   shadow=False,
                   verbose=False):
    """Find the optimal oscillatory voltage at W, over w, for each neuron.
    
    Returns
    ------
    solutions : list((A, phi, sol), ...)
        A list of N 3-tuples 
    """
    if shadow:
        raise NotImplementedError("shadow need to be re-implemented")
    # ---------------------------------------------------------------
    params, w_in, bias_in, sigma = read_modes(mode)
    if not noise:
        sigma = 0

    budget_ref = budget_window(voltage_ref, E + d, w, select=None)

    # least_squares() was struggling with small A, so boost it
    # for param search purposes, then divide it back out in the 
    # problem definition
    rescale = 1e12

    # ---------------------------------------------------------------
    # Define a loss func (closing several locals)
    def est_loss(n, voltage):
        # Select window
        budget = budget_window(voltage, E + d, w, select=None)

        # Get budget terms for opt
        V_free = np.abs(budget_ref['V_free'][n, :])
        V_osc = np.abs(budget['V_osc'][n, :])

        # loss = np.mean(V_free - V_osc)
        loss = V_free - V_osc

        return loss

    # -
    solutions = []
    if select_n is None:
        Ns = list(range(N))
    else:
        Ns = [int(select_n)]

    for i, n in enumerate(Ns):

        def A_problem(p, phi):
            """A new problem for each neuron"""

            A = p[0] / rescale
            if correct_bias:
                bias = bias_in - (A / 2.0)
            else:
                bias = bias_in

            _, _, voltage = adex(
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
                budget=True,
                **params)

            loss = est_loss(n, voltage)

            if verbose:
                budget = budget_window(voltage, E + d, w, select=None)
                V_rest = float(voltage["V_rest"])
                V_osc = np.mean(budget['V_osc'][n, :])
                V_b = float(voltage['V_budget'])
                del_V = np.abs(V_osc) / V_b

                print(
                    ">>> (A {:0.18f}, bias_adj {:0.15f})  ->  (loss {:0.6f}, V_rest {:0.4f}, del_V_osc {:0.4f}))".
                    format(A, bias, np.mean(loss), V_rest, del_V))

            return loss

        # ---------------------------------------------------------------
        # Opt A
        if verbose:
            print(">>> Optimizing A, neuron {}/{}".format(i + 1, len(Ns)))

        # Lst Sq
        p0 = [np.mean([A_0 * rescale, A_max * rescale])]
        bounds = (A_0 * rescale, A_max * rescale)
        if verbose:
            print(">>> p0 {} (rescaled)".format(p0))
            print(">>> bounds {} (rescaled)".format(bounds))

        sol = least_squares(lambda p: A_problem(p, phi_0), p0, bounds=bounds)
        A_hat = sol.x[0]

        # Save
        solutions.append((A_hat / rescale, sol))

    return solutions


def autotune_w(mode,
               w_0,
               rate,
               t=3,
               k=20,
               stim_rate=30,
               seed_stim=1,
               max_mult=2):

    # Create frozen input spikes
    stim_onset = 0.1
    stim_offset = t
    dt = 1e-5
    ns, ts = poisson_impulse(
        t,
        stim_onset,
        stim_offset - stim_onset,
        stim_rate,
        n=k,
        dt=dt,
        seed=seed_stim)

    # -
    def problem(p):
        w = p[0]

        ns_y, ts_y = adex(
            1,
            t,
            ns,
            ts,
            w_in=w,
            bias_in=bias_in,
            sigma=sigma,
            budget=False,
            **params)

        rate_y = ts_y.size / (stim_offset - stim_onset)

        return rate_y - rate

    p0 = [w_0]
    sol = least_squares(problem, p0, bounds=(0, w_0 * max_mult))

    return sol


def autotune_membrane(mode, bias_0, sigma_0, mean, std, t=1):
    # Load cell params
    params, _, _, _ = read_modes(mode)

    # No input spikes
    ns = np.zeros(1)
    ts = np.zeros(1)
    w_in = 0

    # -
    def problem(p):
        bias_in = p[0]
        sigma = p[0]

        vm, _ = shadow_adex(
            1, t, ns, ts, w_in=w_in, bias_in=bias_in, report=None, **params)

        return (np.mean(vm) - mean), (np.std(vm) - std)

    # !
    p0 = [bias_0, sigma_0]
    sol = least_squares(problem, p0)

    return sol
