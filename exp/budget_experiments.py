#!/usr/bin/env python 

import fire
import json
import csv
import os
import numpy as np
from fakespikes.util import spike_window_code
from fakespikes.util import spike_time_code
from fakespikes.util import levenshtein
from fakespikes.util import estimate_communication
from fakespikes.util import precision

import voltagebudget
from voltagebudget.neurons import adex
from voltagebudget.neurons import shadow_adex
from voltagebudget.util import poisson_impulse
from voltagebudget.util import filter_budget
from voltagebudget.util import read_results
from voltagebudget.util import read_stim
from voltagebudget.util import read_args
from voltagebudget.util import read_modes

from platypus.algorithms import NSGAII
from platypus.core import Problem
from platypus.types import Real


def replay(args, stim, results, i, f, save_npy=None, verbose=False):
    """Rerun the results of a budget_experiment"""

    # Load parameters, input, and results
    arg_data = read_args(args)
    stim_data = read_stim(stim)
    results_data = read_results(results)

    # Construct a valid kawrgs for adex()
    exclude = [
        'N', 'time', 'budget', 'report', 'save_args', 'phi', 'w_in', 'A'
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
        w_in=w,
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
            percent_computation,
            percent_communication,
            N=50,
            t=0.8,
            stim_onset=0.6,
            stim_offset=0.7,
            budget_onset=None,
            budget_offset=None,
            stim_rate=20,
            stim_number=20,
            f=0,
            A=0.2e-9,
            phi=np.pi,
            sigma=.1e-9,
            mode='regular',
            reduce_fn='mean',
            M=100,
            fix_w=False,
            fix_A=False,
            fix_phi=False,
            seed_prob=42,
            seed_stim=7525,
            report=None,
            save_only=False,
            verbose=False,
            time_step=1e-4):
    """Optimize using the voltage budget."""
    np.random.seed(seed_prob)

    # --------------------------------------------------------------
    # analysis windows...
    if budget_onset is None:
        budget_onset = stim_onset
    if budget_offset is None:
        budget_offset = stim_offset

    # Get mode
    params, w_in, bias = read_modes(mode)

    # --------------------------------------------------------------
    # Lookup the reduce function
    try:
        reduce_fn = getattr(np, reduce_fn)
    except AttributeError:
        raise ValueError("{} is not a numpy function".format(reduce_fn))

    # -
    if verbose:
        print(">>> Building input.")
    ns, ts = poisson_impulse(
        t,
        stim_onset,
        stim_offset - stim_onset,
        stim_rate,
        dt=1e-5,
        n=stim_number,
        seed=seed_stim)

    if verbose:
        print(">>> {} spikes generated.".format(ns.size))
        print(">>> Saving input.")
    with open("{}_stim.csv".format(name), "wb") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(["ns", "ts"])
        writer.writerows([[nrn, spk] for nrn, spk in zip(ns, ts)])

    # --------------------------------------------------------------
    # Define ideal targt computation (no oscillation)
    # (Make sure and explain this breakdown well in th paper)
    # (it would be an easy point of crit otherwise)
    if verbose:
        print(">>> Creating reference.")

    ns_ref, ts_ref, budget_ref = adex(
        N,
        t,
        ns,
        ts,
        w_in=w_in,
        bias=bias,
        f=0,
        A=0,
        phi=0,
        sigma=sigma,
        seed=seed_prob,
        budget=True,
        report=report,
        save_args="{}_ref_args".format(name),
        **params)

    # --------------------------------------------------------------
    if verbose:
        print(">>> Setting up the problem function.")

    # Move to short math-y variables.
    y_m = reduce_fn(budget_ref['V_comp'])
    z_m = reduce_fn(budget_ref['V_osc'])

    dy = y_m * percent_computation
    dz = z_m * percent_communication

    y_bar = y_m - dy
    z_bar = z_m + dz

    def sim(pars):
        A_p = pars[0]
        phi_p = pars[1]
        w_p = pars[2]

        # Turn off opt on a select
        # parameter? 
        # Resorts to a default.
        if fix_w:
            w_p = w_in
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
            w_in=w_p,
            bias=bias,
            f=f,
            A=A_p,
            phi=phi_p,
            sigma=sigma,
            seed=seed_prob,
            report=report,
            **params)

        # Isolate the analysis window
        budget_o = filter_budget(budget_o['times'], budget_o,
                                 (budget_onset, budget_offset))

        # Reduce the voltages to measures...
        y = reduce_fn(budget_o['V_comp'])
        z = reduce_fn(budget_o['V_osc'])

        # Min. diff between targets and observed
        return np.abs(y - y_bar), np.abs(z - z_bar)

    # --------------------------------------------------------------
    if verbose:
        print(">>> Building problem.")
    problem = Problem(3, 2)
    problem.types[:] = [
        Real(0.0e-12, A), Real(0.0e-12, phi), Real(0.0e-12, w_in)
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
        Ws = [w_in] * M
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
            w_in=w_m,
            bias=bias,
            f=f,
            A=A_m,
            phi=phi_m,
            sigma=sigma,
            budget=True,
            seed=seed_prob,
            report=report,
            **params)

        # -
        times = budget_m['times']
        comm = estimate_communication(
            times,
            ns_m,
            ts_m, (budget_onset, budget_offset),
            coincidence_t=1e-3,
            coincidence_n=20)
        _, prec = precision(ns_m, ts_m, ns_ref, ts_ref, combine=False)

        # -
        communication_scores.append(comm)
        computation_scores.append(np.mean(prec))
        computation_voltages.append(reduce_fn(budget_m['V_comp']))
        communication_voltages.append(reduce_fn(budget_m['V_osc']))

    results["communication_scores"] = communication_scores
    results["computation_scores"] = computation_scores
    results["communication_voltages"] = communication_voltages
    results["computation_voltages"] = computation_voltages

    # --------------------------------------------------------------
    if verbose:
        print(">>> Saving results.")

    keys = sorted(results.keys())
    with open("{}.csv".format(name), "wb") as fi:
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
