#!/usr/bin/env python 

import fire
import csv
import numpy as np
from fakespikes.util import spike_window_code
from fakespikes.util import spike_time_code
from fakespikes.util import levenshtein
from fakespikes.util import estimate_communication
from fakespikes.util import precision

from voltagebudget.neurons import adex
from voltagebudget.neurons import shadow_adex
from voltagebudget.util import poisson_impulse
from voltagebudget.util import filter_budget
from voltagebudget.util import read_results
from voltagebudget.util import read_stim

from platypus.algorithms import NSGAII
from platypus.core import Problem
from platypus.types import Real

global MODES

MODES = {
    'regular': {
        'tau_m': 5e-3,
        'a': -0.5e-9,
        'tau_w': 100e-3,
        'b': 7e-12,
        'E_rheo': -51e-3,
        'delta_t': 2e-3
    },
    'burst': {
        'tau_m': 5e-3,
        'a': -0.5e-9,
        'tau_w': 100e-3,
        'b': 7e-12,
        'E_rheo': -51e-3,
        'delta_t': 2e-3
    },
}


def rerun(stim, results, i, N, t, f, **adex_kwargs):
    # Load stim and results into dict
    stim_data = read_stim(stim)
    results_data = read_results(results)

    # Replay i 
    return adex(
        N,
        t,
        stim_data['ns'],
        stim_data['ts'],
        w_in=stim_data['Ws'][i],
        A=stim_data['As'][i],
        phi=stim_data['Phis'][i],
        f=f,
        budget=True,
        **adex_kwargs)


def forward(name,
            N=50,
            t=0.8,
            stim_onset=0.5,
            stim_offset=0.7,
            budget_onset=0.65,
            budget_offset=0.75,
            w_in=0.8e-3,
            stim_rate=60,
            K=20,
            f=0,
            A=0.2e-9,
            phi=np.pi,
            sigma=.1e-9,
            mode='regular',
            reduce_fn='mean',
            M=10000,
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
    # Lookup the reduce function
    try:
        reduce_fn = getattr(np, reduce_fn)
    except AttributeError:
        raise ValueError("{} is not a numpy function".format(reduce_fn))

    # Set cell-type
    if mode == 'heterogenous':
        raise NotImplementedError("TODO")
    else:
        params = MODES[mode]

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
        print(">>> Saving input.")
    with open("{}_stim.csv".format(name), "wb") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(["n", "t"])
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
    fire.Fire({'forward': forward, 'reverse': reverse})
