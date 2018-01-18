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
from voltagebudget.util import mad
from pyswarm import pso


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
    # ---------------------------------------------------------------
    params, w_in, bias_in, sigma = read_modes(mode)
    if not noise:
        sigma = 0

    budget_ref = budget_window(voltage_ref, E + d, w, select=None)

    # least_squares() was struggling with small A, so boost it
    # for param search purposes, then divide it back out in the 
    # problem definition
    rescale = 1e10

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
    for n in range(N):

        # ---------------------------------------------------------------
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

            loss = est_loss(n, voltage)

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

            loss = est_loss(n, voltage)

            if verbose:
                print(">>> (A {:0.12f}, phi {:0.3f})  ->  (loss {})".format(
                    A / rescale, phi, np.sum(loss)))

            return loss

        # ---------------------------------------------------------------

        # Opt phi
        if verbose:
            print(">>> Optimizing phi, neuron {}/{}.".format(n + 1, N))

        p0 = [phi_0]
        bounds = (0, np.pi)
        sol = least_squares(
            lambda p: phi_problem(p, A_0 * rescale), p0, bounds=bounds)

        phi_hat = sol.x[0]

        # Opt A
        if verbose:
            print(">>> Optimizing A, neuron {}/{}.".format(n + 1, N))

        p0 = [A_0 * rescale]
        bounds = (0, A_max * rescale)
        sol = least_squares(lambda p: A_problem(p, phi_hat), p0, bounds=bounds)

        A_hat = sol.x[0]

        solutions.append((A_hat / rescale, phi_hat, sol))

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
