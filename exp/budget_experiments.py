import fire
import numpy as np
from fakespikes.util import spike_window_code
from fakespikes.util import spike_time_code
from fakespikes.util import levenstien
from voltagebudget.neurons import adex
from voltagebudget.neurons import shadow_adex
from voltagebudget.util import poisson_impulse

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


def foward(name,
           t,
           t_stim,
           n,
           w_in,
           w_sigma,
           f=0,
           A=1e-3,
           phi=0,
           mode='regular',
           seed_value=42,
           time_step=1e-4):
    """Optimize using voltage budget."""
    np.random.seed(seed_value)

    if mode == 'heterogenous':
        raise NotImplementedError("TODO")
    else:
        params = MODES[mode]

    # IN
    w = 20e-3
    a = 60
    n = 20
    ns, ts = util.poisson_impulse(t, t_stim, w, a, n, seed=7425)

    # Define individual neurons (by weight)
    w_ins = np.random.normal(w_in, w_sigma, N)
    w_ins[w_ins < 0] = 0  # Safety

    # Define ideal targt computation (no oscillation)
    # (Make sure and explain this breakdown well in th paper)
    # (it would be an easy point of crit otherwise)
    ns_opt, ts_opt, budget_opt = adex(t, ns, ts, f=0, A=0, phi=0, **params)

    for w in w_ins:
        # !
        ns_osc, ts_osc, budget_osc = adex(
            t, ns, ts, f=f, A=A, phi=phi, **params)

        # Process budget

        # Score


def reverse():
    """Optimize using ISI."""
    pass


if __name__ == "__main__":
    fire.Fire({'forward': forward, 'reverse': reverse})
