import fire
import numpy as np
from fakespikes.util import spike_window_code
from fakespikes.util import spike_time_code
from fakespikes.util import levenstien
from fakespikes.util import estimate_communication
from fakespikes.util import precision

from voltagebudget.neurons import adex
from voltagebudget.neurons import shadow_adex
from voltagebudget.util import poisson_impulse
from voltagebudget.util import filter_budget

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
        'Erheo': -51e-3,
        'delta_t': 2e-3
    },
    'burst': {
        'tau_m': 5e-3,
        'a': -0.5e-9,
        'tau_w': 100e-3,
        'b': 7e-12,
        'Erheo': -51e-3,
        'delta_t': 2e-3
    },
}


def forward(name,
            t=0.8,
            stim_onset=0.5,
            stim_offset=0.7,
            budget_onset=0.4,
            budget_offset=0.5,
            w_in=0.8e-3,
            stim_rate=60,
            N=20,
            f=0,
            A=1e-3,
            phi=np.pi,
            sigma=0,
            mode='regular',
            reduce_fn='mean',
            M=10000,
            fix_w=False,
            fix_A=False,
            fix_phi=False,
            fix_f=False,
            seed_prob=42,
            seed_stim=7525,
            time_step=1e-4):
    """Optimize using the voltage budget."""
    np.random.seed(seed_prob)

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

    # IN
    ns, ts = util.poisson_impulse(
        t,
        stim_onset,
        stim_offset - stim_onset,
        stim_rate,
        n=20,
        seed=seed_stim)

    # --------------------------------------------------------------
    # Define ideal targt computation (no oscillation)
    # (Make sure and explain this breakdown well in th paper)
    # (it would be an easy point of crit otherwise)
    ns_ref, ts_ref = adex(
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
        **params)

    # Setup the problem,
    # which is closed in forward()
    def sim(pars):
        A_p = pars[0]
        phi_p = pars[1]
        f_p = pars[2]
        w_p = pars[3]

        # Turn off opt on a select
        # parameter? 
        # Resorts to a default.
        if fix_w:
            w_p = w_in
        if fix_A:
            A_p = A
        if fix_f:
            f_p = f
        if fix_phi:
            phi_p = phi

        # Run N simulations for mode
        # differing only by noise?
        comp = []
        osc = []
        for n in range(N):
            ns_o, ts_o, budget_o = adex(
                t,
                ns,
                ts,
                w_in=w_p,
                f=f_p,
                A=A_p,
                phi=phi_p,
                sigma=sigma,
                seed=seed_prob + n,
                **params)

            budget_o = filter_budget(budget_o, budget_o['times'],
                                     (budget_onset, budget_offset))

            comp.append(budget_o['V_comp'])
            osc.append(budget_o['V_osc'])

        comp = reduce_fn(np.concatenate(comp))
        osc = reduce_fn(np.concatenate(osc))

        return -comp, -osc

    # --------------------------------------------------------------
    problem = Problem(4, 2)
    problem.types[:] = [
        Real(0.0, A), Real(0.0, phi), Real(0, f), Real(0.0, w_in)
    ]
    problem.function = sim

    algorithm = NSGAII(problem)
    algorithm.run(M)

    results = dict(
        V_comp=[s.objectives[0] for s in algorithm.result],
        V_osc=[s.objectives[1] for s in algorithm.result],
        As=[s.variables[0] for s in algorithm.result],
        Phis=[s.variables[1] for s in algorithm.result],
        Fs=[s.variables[2] for s in algorithm.result],
        Ws=[s.variables[3] for s in algorithm.result])

    # --------------------------------------------------------------
    # Score spiking from the optimal (?) budgets
    communication_scores = []
    precision_scores = []

    for n in range(N):
        A_n = results['As'][i]
        phi_n = results['Phis'][i]
        f_n = results['Fs'][i]
        w_n = results['Ws'][i]

        ns_n, ts_n = adex(
            t,
            ns,
            ts,
            w_in=w_in,
            f=0,
            A=0,
            phi=0,
            sigma=sigma,
            budget=False,
            seed=seed_prob + n,
            **params)

        comm = estimate_communication(
            times, ns, ts, window, coincidence_t=1e-3, coincidence_n=20)
        _, prec = precision(ns_n, ts_n, ns_ref, ts_ref, combine=True)

        communication_scores.append(comm)
        precision_scores.append(prec)

    results["communication_scores"] = communication_scores
    results["precision_scores"] = precision_scores

    # --------------------------------------------------------------
    # Fin!
    # Write
    keys = sorted(results.keys())
    with open("{}.csv".format(name), "wb") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(keys)
        writer.writerows(zip(* [results[key] for key in keys]))

    return results


def reverse():
    """Optimize using ISI."""
    pass


if __name__ == "__main__":
    fire.Fire({'forward': forward, 'reverse': reverse})
