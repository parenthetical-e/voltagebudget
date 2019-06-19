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
from voltagebudget.util import mad

# # -
# def _run(ts, target, fn, verbose=False):
#     # init
#     initial_variances = []
#     target_variances = []
#     obs_variances = []
#     errors = []

#     # run of changes
#     for c in changes:
#         if verbose:
#             print(">>> Running c {}".format(c))

#         obs_variances.append(obs)
#         initial_variances.append(initial)
#         target_variances.append(target)
#         errors.append(err)

#     results = [obs_variances, initial_variances, target_variances, errors]

#     return results


def optimal(name,
            stim,
            var_target,
            alg='max_deviants',
            n_samples=100,
            verbose=False,
            save_spikes=False,
            save_only=False):
    """Run an 'optimal' PLC experiment"""

    # --------------------------------------------------------------
    if verbose:
        print(">>> Looking up the alg.")

    if alg == 'max_deviants':
        fn = max_deviant
    elif alg == 'left_deviants':
        fn = lambda ts, intial, target: max_deviant(ts, intial, target, side='left')
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

    var_ref = mad(ts)

    # --------------------------------------------------------------
    # Run plc!
    if verbose:
        print(">>> Running {}.".format(alg))

    initial_variances = []
    target_variances = []
    obs_variances = []
    errors = []
    changes = np.linspace(var_ref, var_target, n_samples + 1)
    for var_c in changes:
        _, target, adjusted, ts_adjusted = fn(ts, var_ref, var_c)
        error = mae(ts, ts_adjusted)

        initial_variances.append(var_ref)
        target_variances.append(target)
        obs_variances.append(adjusted)
        errors.append(error)

        if save_spikes:
            ts_name = "{}_target{}_spikes".format(name, var_c)
            write_spikes(ts_name, ns, ts_adjusted)

    # --------------------------------------------------------------
    if verbose:
        print(">>> Saving results.")

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
        writer.writerows(zip(*[results[key] for key in keys]))

    # -
    # If running in a CL, returns are line noise?
    if not save_only:
        return results
    else:
        return None