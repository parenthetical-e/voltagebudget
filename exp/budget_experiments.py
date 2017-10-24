import fire
import numpy as np
from fakespikes.util import spike_window_code
from fakespikes.util import spike_time_code
from fakespikes.util import levenstien
from voltagebudget.neurons import adex
from voltagebudget.neurons import shadow_adex
from voltagebudget.util import poisson_impulse
from voltagebudget.util import filter_budget

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
            w_sigma=0.08e-3,
            stim_rate=60,
            N=20,
            f=0,
            A=1e-3,
            phi=0,
            sigma=0,
            mode='regular',
            seed_prob=42,
            seed_stim=7525,
            time_step=1e-4):
    """Optimize using voltage budget."""
    np.random.seed(seed_value)

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

    # Define individual neurons (by weight)
    w_ins = np.random.normal(w_in, w_sigma, N)
    w_ins[w_ins < 0] = 0.01e-9  # Safety

    # Define ideal targt computation (no oscillation)
    # (Make sure and explain this breakdown well in th paper)
    # (it would be an easy point of crit otherwise)
    ns_c, ts_c, budget_c = adex(
        t,
        ns,
        ts,
        w_in=w_in,
        f=0,
        A=0,
        phi=0,
        sigma=sigma,
        seed=seed_prob,
        **params)

    # Use C budget and shadow osc to find the ideal osc
    # In the budget_onset, budget_offset window
    budget_w = filter_budget(budget_c, budget_c['times'],
                             (budget_onset, budget_offset))

    V_free = budget_w['V_free']
    E_thresh = budget_w['E_thresh']

    # If the whole system spiked, end early.
    if np.min(V_free) >= E_thresh:
        return 0, 0, 0

    # Otherwise setup the problem
    def problem(pars):
        A = pars[0]
        phi = pars[1]
        f = pars[2]
        w = pars[3]
        if w < 0:
            w = w_in  # Default value 

        free = 0
        for n in range(N):
            # !
            ns_o, ts_o, budget_o = adex(
                t,
                ns,
                ts,
                w_in=w,
                f=f,
                A=A,
                phi=phi,
                seed=seed_prob + n,
                **params)

            budget_o = filter_budget(budget_o, budget_o['times'],
                                     (budget_onset, budget_offset))

        return -comp, -osc


def reverse():
    """Optimize using ISI."""
    pass


if __name__ == "__main__":
    fire.Fire({'forward': forward, 'reverse': reverse})
