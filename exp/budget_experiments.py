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

from platypus.algorithms import NSGAII
from platypus.core import Problem
from platypus.types import Real


def replay(args, stim, results, i, save_npy=None, verbose=False):
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
        budget=True,
        report=None,
        **kwargs)

    if save_npy is not None:
        np.savez(save, ns=ns, ts=ts, budget=budget)
    else:
        return ns, ts, budget


def forward(name,
            N=50,
            t=0.8,
            stim_onset=0.6,
            stim_offset=0.7,
            budget_onset=0.6,
            budget_offset=0.7,
            w_in=1e-9,
            stim_rate=20,
            K=20,
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
    # Read in modes:
    json_path = os.path.join(
        os.path.split(voltagebudget.__file__)[0], 'modes.json')
    with open(json_path, 'r') as data_file:
        modes = json.load(data_file)

    # And select one...
    params = modes[mode]

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
        n=K,
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
    ns_ref, ts_ref = adex(
        N,
        t,
        ns,
        ts,
        w_in=w_in,
        f=0,
        A=0,
        phi=0,
        sigma=sigma,
        seed=seed_prob,
        budget=False,
        report=report,
        save_args="{}_ref_args".format(name),
        **params)

    if verbose:
        print(">>> Reference times {}".format(ts_ref[:5]))

    # Setup the problem,
    # which is closed in forward()
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
        comp = []
        osc = []

        ns_o, ts_o, budget_o = adex(
            N,
            t,
            ns,
            ts,
            w_in=w_p,
            f=f,
            A=A_p,
            phi=phi_p,
            sigma=sigma,
            seed=seed_prob,
            report=report,
            **params)

        if verbose:
            print(">>> Osc. times {}".format(ts_o[:5]))

        budget_o = filter_budget(budget_o['times'], budget_o,
                                 (budget_onset, budget_offset))

        comp.append(budget_o['V_comp'])
        osc.append(budget_o['V_osc'])

        comp = reduce_fn(np.vstack(comp))
        osc = reduce_fn(np.vstack(osc))

        return -comp, -osc

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

    results = dict(
        V_comp=[s.objectives[0] for s in algorithm.result],
        V_osc=[s.objectives[1] for s in algorithm.result],
        As=[s.variables[0] for s in algorithm.result],
        Phis=[s.variables[1] for s in algorithm.result],
        Ws=[s.variables[2] for s in algorithm.result])

    # --------------------------------------------------------------
    if verbose:
        print(">>> Analyzing results.")

    communication_scores = []
    precision_scores = []

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
            f=f,
            A=A_m,
            phi=phi_m,
            sigma=sigma,
            budget=True,
            seed=seed_prob,
            report=report,
            **params)

        times = budget_m['times']
        comm = estimate_communication(
            times,
            ns_m,
            ts_m, (budget_onset, budget_offset),
            coincidence_t=1e-3,
            coincidence_n=20)
        _, prec = precision(ns_m, ts_m, ns_ref, ts_ref, combine=False)

        communication_scores.append(comm)
        precision_scores.append(np.mean(prec))

    results["communication_scores"] = communication_scores
    results["precision_scores"] = precision_scores

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
    """Optimize using ISI."""
    pass


if __name__ == "__main__":
    fire.Fire({'forward': forward, 'reverse': reverse, 'replay': replay})
