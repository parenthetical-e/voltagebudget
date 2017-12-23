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


def sweep_power(name,
                N=50,
                t=0.4,
                budget_delay=-10e-3,
                budget_width=2e-3,
                combine_budgets=False,
                budget_reduce_fn='mean',
                stim_number=40,
                stim_onset=0.2,
                stim_offset=0.250,
                stim_rate=8,
                coincidence_t=1e-3,
                f=10,
                A_intial=0.0e-9,
                A_final=.05e-9,
                M=20,
                phi=np.pi,
                mode='regular',
                report=None,
                save_only=False,
                verbose=False,
                seed_stim=1,
                time_step=1e-5):

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
        seed=seed_stim,
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
    As = np.linspace(A_intial, A_final, M)
    if verbose:
        print(">>> Sweeping over {} powers.".format(len(As)))

    # -
    communication_scores = []
    computation_scores = []
    communication_voltages = []
    computation_voltages = []
    for A in As:
        ns_o, ts_o, budget_o = adex(
            N,
            t,
            ns,
            ts,
            w_max=w_max,
            bias=bias,
            f=f,
            A=A,
            phi=phi,
            sigma=sigma,
            seed=seed_stim,
            budget=True,
            report=report,
            save_args=None,
            time_step=time_step,
            **params)

        # Isolate the reference analysis window
        voltages_o = filter_voltages(
            budget_o,
            ns_first,
            ts_first,
            budget_delay=budget_delay,
            budget_width=budget_width,
            combine=combine_budgets)

        # --------------------------------------------------------------
        # Extract stats, both voltages and scores
        comm = estimate_communication(
            ns_o,
            ts_o, (stim_onset, stim_offset),
            coincidence_t=coincidence_t,
            time_step=time_step)

        _, prec = precision(
            ns_o, ts_o, ns_ref, ts_ref, combine=combine_budgets)

        communication_scores.append(comm)
        computation_scores.append(np.mean(prec))

        # --------------------------------------------------------------
        V_comp = budget_reduce_fn(voltages_o['V_comp'])
        V_comm = budget_reduce_fn(voltages_o['V_osc'])
        computation_voltages.append(V_comp)
        communication_voltages.append(V_comm)

    results = {}
    results["power"] = As
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
