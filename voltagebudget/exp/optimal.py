import os
import csv
import numpy as np

import voltagebudget

from voltagebudget.util import read_stim
from voltagebudget.util import write_spikes
from voltagebudget.plc import max_deviant
from voltagebudget.plc import uniform
from voltagebudget.plc import coincidence

from voltagebudget.util import mae


# -
def _run(ts, changes, fn, save_ts=False):
    # init
    initial_variances = []
    target_variances = []
    obs_variances = []
    errors = []

    if save_ts:
        ts_opts = []

    # run of changes
    for c in changes:
        initial, target, obs, ts_opt = fn(ts, c)
        err = mae(ts, ts_opt)

        obs_variances.append(obs)
        initial_variances.append(initial)
        target_variances.append(target)
        errors.append(err)

        if save_ts:
            ts_opts.append(ts_opt)

    # Build the result
    results = [obs_variances, initial_variances, target_variances, errors]
    if save_ts:
        results.append(ts_opts)

    return results


def optimal(name,
            stim,
            p_0,
            p_max,
            alg='max_deviants',
            n_samples=100,
            verbose=False,
            save_spikes=False,
            save_only=False):
    """Run a PLC experiment"""

    # --------------------------------------------------------------
    if verbose:
        print(">>> Looking up the alg.")

    if alg == 'max_deviants':
        fn = max_deviant
    elif alg == 'left_deviants':
        fn = lambda x, p: max_deviant(x, p, side='left')
    elif alg == 'uniform':
        fn = uniform
    elif alg == 'coincidence':
        fn = coincidence
    else:
        raise ValueError("alg type was unknown.")

    # --------------------------------------------------------------
    if verbose:
        print(">>> Importing stimulus from {}.".format(stim))
    stim_data = read_stim(stim)
    ns = np.asarray(stim_data['ns'])
    ts = np.asarray(stim_data['ts'])

    # --------------------------------------------------------------
    # Run plc!
    if verbose:
        print(">>> Running {}.".format(alg))
    changes = np.linspace(p_0, p_max, n_samples)

    if save_spikes:
        (obs_variances, initial_variances, target_variances, errors,
         ts_cs) = _run(
             ts, changes, fn, save_ts=save_spikes)
    else:
        (obs_variances, initial_variances, target_variances, errors) = _run(
            ts, changes, fn)

    # --------------------------------------------------------------
    if verbose:
        print(">>> Saving results.")

    # -
    results = {
        'percent_change': changes,
        'obs_variances': obs_variances,
        'initial_variances': initial_variances,
        'target_variances': target_variances,
        'errors': errors
    }

    # and write them out.
    keys = sorted(results.keys())
    with open("{}.csv".format(name), "w") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(keys)
        writer.writerows(zip(* [results[key] for key in keys]))

    # -
    if save_spikes:
        if verbose:
            print("Saving spikes.")

        for c, ts_c in zip(changes, ts_cs):
            ts_name = "{}_c{}".format(name, c)
            write_spikes(ts_name, ns, ts_c)

    # -
    # If running in a CL, returns are line noise?
    if not save_only:
        return results
    else:
        return None