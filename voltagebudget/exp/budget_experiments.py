#!/usr/bin/env python

import fire
import json
import csv
import os
import numpy as np

import voltagebudget
from voltagebudget.neurons import adex
from voltagebudget.neurons import shadow_adex
from voltagebudget.util import poisson_impulse
from voltagebudget.util import read_results
from voltagebudget.util import read_stim
from voltagebudget.util import read_args
from voltagebudget.util import read_modes

from voltagebudget.budget import filter_voltages
from voltagebudget.budget import locate_firsts
from voltagebudget.budget import locate_peaks
from voltagebudget.budget import estimate_communication
from voltagebudget.budget import precision

from platypus.algorithms import NSGAII
from platypus.core import Problem
from platypus.types import Real

from scipy.optimize import least_squares


def autotune_w(mode,
               w_0,
               rate,
               t=3,
               k=20,
               stim_rate=30,
               seed_stim=1,
               max_mult=2):
    # Load cell params
    params, _, bias, sigma = read_modes(mode)

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
            w_max=w,
            bias=bias,
            sigma=sigma,
            report=None,
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
    w_max = 0

    # -
    def problem(p):
        bias = p[0]
        sigma = p[0]

        vm, _ = shadow_adex(
            1, t, ns, ts, w_max=w_max, bias=bias, report=None, **params)

        return (np.mean(vm) - mean), (np.std(vm) - std)

    # !
    p0 = [bias_0, sigma_0]
    sol = least_squares(problem, p0)

    return sol


def replay(args, stim, results, i, f, save_npy=None, verbose=False):
    """Rerun the results of a budget_experiment"""

    # Load parameters, input, and results
    arg_data = read_args(args)
    stim_data = read_stim(stim)
    results_data = read_results(results)

    # Construct a valid kawrgs for adex()
    exclude = [
        'N', 'time', 'budget', 'report', 'save_args', 'phi', 'w_max', 'A'
    ]
    kwargs = {}
    for k, v in arg_data.items():
        if k not in exclude:
            kwargs[k] = v

    w = results_data['Ws'][i]
    A = results_data['As'][i]
    phi = results_data['Phis'][i]

    # drop f=0
    kwargs.pop("f", None)

    # Replay row i results
    if verbose:
        print(">>> Replaying with optimal parameters w:{}, A:{}, phi:{}".
              format(w, A, phi))
        print(">>> Default paramerers")
        print(kwargs)

    ns, ts, budget = adex(
        arg_data['N'],
        arg_data['time'],
        np.asarray(stim_data['ns']),
        np.asarray(stim_data['ts']),
        w_max=w,
        A=A,
        phi=phi,
        f=f,
        budget=True,
        report=None,
        **kwargs)

    if save_npy is not None:
        np.savez(save, ns=ns, ts=ts, budget=budget)
    else:
        return ns, ts, budget


def forward(name,
            N=50,
            t=0.9,
            budget_bias=0,
            budget_delay=-10e-3,
            budget_width=2e-3,
            combine_budgets=False,
            budget_reduce_fn='mean',
            stim_onset=0.6,
            stim_offset=0.8,
            stim_rate=12,
            stim_number=50,
            coincidence_t=2e-3,
            coincidence_n=20,
            f=0,
            A=0.2e-9,
            phi=np.pi,
            mode='regular',
            M=100,
            fix_w=False,
            fix_A=False,
            fix_phi=False,
            seed_prob=42,
            seed_stim=7525,
            report=None,
            save_only=False,
            verbose=False,
            time_step=2.5e-5):
    """Optimize using the voltage budget."""
    np.random.seed(seed_prob)

    # --------------------------------------------------------------
    # Get mode
    params, w_max, bias, sigma = read_modes(mode)

    # --------------------------------------------------------------
    # Lookup the reduce function
    try:
        budget_reduce_fn = getattr(np, budget_reduce_fn)
    except AttributeError:
        raise ValueError("{} is not a numpy function".format(budget_reduce_fn))

    # --------------------------------------------------------------
    if verbose:
        print(">>> Building input.")
    ns, ts = poisson_impulse(
        t,
        stim_onset,
        stim_offset - stim_onset,
        stim_rate,
        dt=0.5e-4,
        n=stim_number,
        seed=seed_stim)

    if verbose:
        print(">>> {} spikes generated.".format(ns.size))
        print(">>> Saving input.")
    with open("{}_stim.csv".format(name), "w") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(["ns", "ts"])
        writer.writerows([[nrn, spk] for nrn, spk in zip(ns, ts)])

    # --------------------------------------------------------------
    # Define target computation (i.e., no oscillation)
    # (TODO Make sure and explain this breakdown well in th paper)
    if verbose:
        print(">>> Creating reference.")

    ns_ref, ts_ref, budget_ref = adex(
        N,
        t,
        ns,
        ts,
        w_max=w_max,
        bias=bias,
        f=0,
        A=0,
        phi=0,
        sigma=sigma,
        seed=seed_prob,
        budget=True,
        report=report,
        save_args="{}_ref_args".format(name),
        time_step=time_step,
        **params)

    if ns_ref.size == 0:
        raise ValueError("The reference model didn't spike.")

    # Isolate the reference analysis window
    ns_first, ts_first = locate_firsts(ns_ref, ts_ref, combine=combine_budgets)
    voltages_ref = filter_voltages(
        budget_ref,
        ns_first,
        ts_first,
        budget_delay=budget_delay,
        budget_width=budget_width,
        combine=combine_budgets)

    # --------------------------------------------------------------
    if verbose:
        print(">>> Setting up the problem function.")

    def sim(pars):
        A_p = pars[0]
        phi_p = pars[1]
        w_p = pars[2]

        # Turn off opt on a select
        # parameter?
        # Resorts to a default.
        if fix_w:
            w_p = w_max
        if fix_A:
            A_p = A
        if fix_phi:
            phi_p = phi

        # Run N simulations for mode
        # differing only by noise?
        ns_o, ts_o, budget_o = adex(
            N,
            t,
            ns,
            ts,
            w_max=w_p,
            bias=bias,
            f=f,
            A=A_p,
            phi=phi_p,
            sigma=sigma,
            seed=seed_prob,
            report=report,
            time_step=time_step,
            **params)

        # -
        # Locate either spikes or vm max values in the stim_window
        if ns_o.size > 0:
            ns_first, ts_first = locate_firsts(
                ns_o, ts_o, combine=combine_budgets)
        else:
            ns_first, ts_first = locate_peaks(
                budget_o, stim_onset, stim_offset, combine=combine_budgets)

        # Extract voltages based on spikes/max
        voltages_ref = filter_voltages(
            budget_o,
            ns_first,
            ts_first,
            budget_delay=budget_delay,
            budget_width=budget_width,
            combine=combine_budgets)

        # Reduce the voltages
        y = budget_o['V_comp']
        z = budget_o['V_osc']
        y = budget_reduce_fn(y)
        z = budget_reduce_fn(z)

        return (-y + budget_bias, -z - budget_bias)

    # --------------------------------------------------------------
    if verbose:
        print(">>> Building problem.")
    problem = Problem(3, 2)
    problem.types[:] = [
        Real(0.0e-12, A), Real(0.0e-12, phi), Real(0.0e-12, w_max)
    ]
    problem.function = sim
    algorithm = NSGAII(problem)

    if verbose:
        print(">>> Running problem.")
    algorithm.run(M)

    # Build results
    if verbose:
        print(">>> Building results.")
    results = dict(
        Opt_y=[s.objectives[0] for s in algorithm.result],
        Opt_z=[s.objectives[1] for s in algorithm.result])

    if not fix_A:
        As = [s.variables[0] for s in algorithm.result]
    else:
        As = [A] * M
    if not fix_phi:
        Phis = [s.variables[1] for s in algorithm.result]
    else:
        Phis = [phi] * M
    if not fix_w:
        Ws = [s.variables[2] for s in algorithm.result]
    else:
        Ws = [w_max] * M

    results['As'] = As
    results['Phis'] = Phis
    results['Ws'] = Ws

    # --------------------------------------------------------------
    if verbose:
        print(">>> Analyzing results.")

    communication_scores = []
    computation_scores = []
    communication_voltages = []
    computation_voltages = []
    for m in range(M):
        A_m = results['As'][m]
        phi_m = results['Phis'][m]
        w_m = results['Ws'][m]

        ns_m, ts_m, budget_m = adex(
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
            **params)

        # -
        times = budget_m['times']
        comm = estimate_communication(
            times,
            ns_m,
            ts_m, (stim_onset, stim_offset),
            coincidence_t=coincidence_t,
            coincidence_n=coincidence_n)
        communication_scores.append(comm)

        # -
        _, prec = precision(
            ns_m, ts_m, ns_ref, ts_ref, combine=combine_budgets)
        computation_scores.append(np.mean(prec))

        # -
        if ns_m.size > 0:
            ns_first, ts_first = locate_firsts(
                ns_m, ts_m, combine=combine_budgets)
        else:
            ns_first, ts_first = locate_peaks(
                budget_m, stim_onset, stim_offset, combine=combine_budgets)

        voltages_m = filter_voltages(
            budget_m,
            ns_first,
            ts_first,
            budget_delay=budget_delay,
            budget_width=budget_width,
            combine=combine_budgets)

        comp = budget_reduce_fn(voltages_m['V_comp'])
        comm = budget_reduce_fn(voltages_m['V_osc'])

        computation_voltages.append(comp)
        communication_voltages.append(comm)

    # -
    results["communication_scores"] = communication_scores
    results["computation_scores"] = computation_scores
    results["communication_voltages"] = communication_voltages
    results["computation_voltages"] = computation_voltages

    # --------------------------------------------------------------
    if verbose:
        print(">>> Saving results.")

    keys = sorted(results.keys())
    with open("{}.csv".format(name), "w") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(keys)
        writer.writerows(zip(* [results[key] for key in keys]))

    if not save_only:
        return results


def reverse():
    """Optimize using metrics not voltages."""
    pass


if __name__ == "__main__":
    fire.Fire({'forward': forward, 'reverse': reverse, 'replay': replay})
