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
        np.savez(save_npy, ns=ns, ts=ts, budget=budget)
    else:
        return ns, ts, budget
