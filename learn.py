from joblib import Parallel, delayed
from fakespikes.util import bin_times
import numpy as np


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


def lif_brute(time, ns, ts, ts_y, w_range, bias_range, num=10, n_jobs=10):
    # Grid search
    ws = np.linspace(*w_range, num=num)
    bs = np.linspace(*bias_range, num=num)
    results = Parallel(n_jobs=n_jobs)(delayed(lif)(
        time, 1, ns, ts, w_in, bias=bias, budget=False) for w_in in ws)

    # Extract spike times
    ts_hats = [r[1] for r in results]

    # Est. min error
    # Loop over y_hats, comparing to ts_y
    errors = []
    for i, ts_hat in enumerate(ts_hats):
        errors_i = []
        for t in ts_hat:
            # Find closests to ts_y
            idx = (np.abs(ts_y - t)).argmin()
            errors_i.append(np.abs(ts_y[idx] - t))
        errors.append(np.mean(errors_i))

    # Return best weight
    lowest = np.argmin(errors)

    return ws[lowest], bs[lowest]


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
