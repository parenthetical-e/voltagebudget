from itertools import product
import numpy as np

from joblib import Parallel, delayed
from fakespikes.util import bin_times
from voltagebudget.neurons import lif, adex


def coincidence_detection(ts, k=20, a_tol=1e-3):
    ts = np.sort(ts)
    n = len(ts)

    # Coincidences are spikes within a_tol
    coincidences = []

    i = 0
    while i < n:
        t = ts[i]

        # Find coincidences, and count them
        diff = np.abs(t - ts)
        C_index = diff <= a_tol
        C = np.sum(C_index)

        if C >= k:
            # If there is enough, save the time t
            coincidences.append(t)

            # Advance t by C, or just 1
            i += (C + 1)
        else:
            i += 1

    return coincidences


def lif_brute(time,
              ns,
              ts,
              ts_y,
              w_range,
              bias_range,
              num=10,
              n_jobs=8,
              diagnostic=False):
    # Grid search
    try:
        w1, w2 = w_range
        ws = np.linspace(w1, w2, num=num)
    except ValueError:
        ws = [
            w_range,
        ]

    try:
        b1, b2 = b_range
        bs = np.linspace(b1, b2, num=num)
    except ValueError:
        bs = [
            b_range,
        ]

    params = list(product(ws, bs))

    results = Parallel(
        n_jobs=n_jobs, verbose=1)(delayed(lif)(
            time, 1, ns, ts, w_in=w_in, bias=bias, budget=False, report=None)
                                  for (w_in, bias) in params)

    # Extract spike times
    ts_hats = [r[1] for r in results]

    # Est. min error.
    # Loop over y_hats, comparing to ts_y
    errors = []
    for i, ts_hat in enumerate(ts_hats):
        # Punish and skip empty lists
        if ts_hat.size == 0:
            errors.append(999999)
            continue

        # Local errors
        errors_i = []

        # Find closests to ts_y
        for t in ts_hat:
            idx = (np.abs(ts_y - t)).argmin()
            errors_i.append(np.abs(ts_y[idx] - t))

        # Global mean error for these params
        errors.append(np.mean(errors_i))

    # Handle nan
    errors = np.asarray(errors)
    errors[np.logical_not(np.isfinite(errors))] = 9999999

    # Return best weight
    lowest = np.argmin(errors)

    # Predict with best
    w_best, b_best = params[lowest]
    ts_best = lif(time, 1, ns, ts, w_in=w_best, bias=b_best, budget=False)[1]

    if diagnostic:
        details = {"lowest": lowest, "errors": errors}
        return w_best, b_best, details

    return w_best, b_best, ts_best


def coincidence(args,
                nrn,
                readout,
                step_k,
                step_l,
                criterion,
                target_fn,
                max_iteration=1000):
    t = 0
    i = 0
    error = 9999999999

    # Does k evenly divide l?
    if np.isclose(step_l % step_k, 0):
        raise ValueError("step_k must evenly divide step_l")

    # How many steps of k in l?
    m = int(step_l / step_k)

    while error < criterion:
        if i > max_iteration:
            raise Exception(
                "Model did not converge! `max_interation` was exceeded.")

        ns, ts = nrn(t, *args, dt=step_k)
        y = readout(t, ns, ts)
        y_bar = target_fn(t, ns, ts)

        error = y_bar - y

        # RLS

        # Counters
        t += step_l
        i += 1
