import json
import csv
import os
import numpy as np

import voltagebudget

from scipy.optimize import least_squares

from voltagebudget.neurons import adex
from voltagebudget.neurons import shadow_adex

from voltagebudget.util import poisson_impulse
from voltagebudget.util import read_results
from voltagebudget.util import read_stim
from voltagebudget.util import read_args
from voltagebudget.util import read_modes

from voltagebudget.util import filter_voltages
from voltagebudget.util import budget_window
from voltagebudget.util import locate_firsts
from voltagebudget.util import locate_peaks
from voltagebudget.util import estimate_communication
from voltagebudget.util import precision


def autotune_V_osc(N,
                   t,
                   E,
                   d,
                   ns,
                   ts,
                   voltage_ref,
                   w=2e-3,
                   A_0=0.1e-9,
                   A_max=0.5e-9,
                   phi_0=0,
                   f=8,
                   mode='regular',
                   noise=False,
                   seed_value=42,
                   shadow=True,
                   verbose=False):
    """Find the optimal oscillatory voltage at W, over w, for each neuron.
    
    Returns
    ------
    solutions : list((A, phi, sol), ...)
        A list of N 3-tuples 
    """
    # -
    params, w_in, bias_in, sigma = read_modes(mode)
    if not noise:
        sigma = 0

    budget_ref = budget_window(voltage_ref, E + d, w, select=None)

    # least_squares() was struggling with small A, so boost it
    # for param search purposes, then divide it back out in the 
    # problem definition
    rescale = 1e10

    p0 = (A_0 * rescale, phi_0)
    bounds = ([0, 0], [A_max * rescale, np.pi])

    # -
    solutions = []
    for n in range(N):

        def problem(p):
            """A new problem for each neuron"""
            A = p[0] / rescale
            phi = p[1]

            # Run into the shadow! realm!
            if shadow:
                voltage = shadow_adex(
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
                    **params)
            else:
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

            # Select window
            budget = budget_window(voltage, E + d, w, select=None)

            # Get budget terms for opt
            V_free = np.abs(np.mean(budget_ref['V_free'][n, :]))
            V_osc = np.abs(np.mean(budget['V_osc'][n, :]))

            loss = V_free - V_osc

            if verbose:
                print(
                    ">>> (A {:0.12f}, phi {:0.3f})  ->  (V_free {:0.5f}, V_osc {:0.5f} loss {:0.5f})".
                    format(A, phi, V_free, V_osc, loss))

            return loss

        # !
        if verbose:
            print(">>> Optimizing neuron {}/{}.".format(n + 1, N))

        sol = least_squares(problem, p0, bounds=bounds, ftol=1e-4)
        A_opt, phi_opt = sol.x

        solutions.append((A_opt / rescale, phi_opt, sol))

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
