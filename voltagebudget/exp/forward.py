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


def forward(name,
            N=50,
            t=0.4,
            budget_bias=0,
            budget_delay=-10e-3,
            budget_width=2e-3,
            combine_budgets=False,
            budget_reduce_fn='mean',
            stim_number=40,
            stim_onset=0.2,
            stim_offset=0.250,
            stim_rate=8,
            coincidence_t=1e-3,
            f=0,
            A=.05e-9,
            phi=np.pi,
            mode='regular',
            M=100,
            fix_w=False,
            seed_prob=42,
            seed_stim=7525,
            report=None,
            save_only=False,
            verbose=False,
            time_step=1e-5):
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
        dt=time_step,
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

        if fix_w:
            w_p = w_max
        else:
            w_p = pars[2]

        if verbose:
            print("(A, phi, w) : ({}, {}, {})".format(A_p, phi_p, w_p))

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

        # Extract voltages are same time points
        # as the ref
        voltages_o = filter_voltages(
            budget_o,
            ns_first,
            ts_first,
            budget_delay=budget_delay,
            budget_width=budget_width,
            combine=combine_budgets)

        # Reduce the voltages
        y = voltages_o['V_comp']
        z = voltages_o['V_osc']
        y = budget_reduce_fn(y)
        z = budget_reduce_fn(z)

        if verbose:
            print("(y, z) : ({}, {})".format(y, z))
        return (y + budget_bias, z - budget_bias)

    # --------------------------------------------------------------
    if verbose:
        print(">>> Building problem.")

    if fix_w:
        problem = Problem(2, 2)
        problem.types[:] = [Real(0.0e-12, A), Real(0.0e-12, phi)]
    else:
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

    # Add vars
    As = [s.variables[0] for s in algorithm.result]
    Phis = [s.variables[1] for s in algorithm.result]

    if fix_w:
        Ws = [w_max] * M
    else:
        Ws = [s.variables[2] for s in algorithm.result]

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
        comm = estimate_communication(
            ns_m,
            ts_m, (stim_onset, stim_offset),
            coincidence_t=coincidence_t,
            time_step=time_step)
        communication_scores.append(comm)

        _, prec = precision(
            ns_m, ts_m, ns_ref, ts_ref, combine=combine_budgets)
        computation_scores.append(np.mean(prec))

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
