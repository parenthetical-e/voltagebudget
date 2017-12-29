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

from voltagebudget.budget import filter_voltages
from voltagebudget.budget import budget_window
from voltagebudget.budget import locate_firsts
from voltagebudget.budget import locate_peaks
from voltagebudget.budget import estimate_communication
from voltagebudget.budget import precision


def autotune_V_osc(N,
                   t,
                   E,
                   d,
                   ns,
                   ts,
                   w=2e-3,
                   A0=0.1e-9,
                   phi0=0,
                   f0=8,
                   mode='regular',
                   max_mult=2,
                   seed_value=42,
                   opt_f=False,
                   verbose=False):
    """Find the optimal oscillatory voltage at W, over w, for each neuron.
    
    Returns
    ------
    solutions : list(sol_n, sol_n+1, ...)
        A list of N least_squares solution objects
    """
    # -
    params, w_in, bias_in, sigma = read_modes(mode)

    # -
    solutions = []
    for n in range(N):
        # Initialize opt params
        if opt_f:
            p0 = (A0, phi0, f0)
            bounds = ([0, 0, 1], [A0 * max_mult, np.pi, 80])
        else:
            p0 = (A0, phi0)
            bounds = ([0, 0], [A0 * max_mult, np.pi])

        def problem(p):
            """A new problem for each neuron"""
            A = p[0]
            phi = p[1]
            if opt_f:
                f = p[2]
            else:
                f = f0

            # Run into the shadow! realm!
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
                seed_value=seed_value,
                **params)

            # Select window
            budget = budget_window(
                voltage, E + d, w, select=None, combine=False)

            # Get budget terms for opt
            V_b = np.abs(np.mean(voltage['V_budget'][n]))
            V_c = np.abs(np.mean(budget['V_comp'][n, :]))
            V_o = np.abs(np.mean(budget['V_osc'][n, :]))

            return V_b - (V_o + V_c)

        # !
        if verbose:
            print(">>> Optimizing neuron {}/{}.".format(n + 1, N))

        sol = least_squares(problem, p0, bounds=bounds)
        solutions.append(sol)

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
