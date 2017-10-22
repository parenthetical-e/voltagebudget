"""Usage: amp.py NAME 
        (--lif | --adex)
        [-w W] [-s S] [-a A] [-f F] [-n N] [-t T]
        [--n_grid NGRID]

Explore oscillation amplitude's effect on communication and computation.

    Arguments:
        NAME    results name (.hdf5)

    Options:
        -h --help               show this screen
        -w W                    average input weight [default: 0.2e-9]
        -s S                    std dev of the input weight [default: 0.5]
        -a A                    maximum oscillation size (amp) [default: 30e-3]
        -f F                    oscillation frequency (Hz) [default: 50]
        -t T                    stim onset time (< 0.2) [default: 0.1]
        -n N                    number of Y neurons [default: 100]
        --n_grid NGRID          N pts. for sampling [0, A] [default: 20]
"""

# %matplotlib inline
# import matplotlib.pyplot as plt
from __future__ import division

import csv

import numpy as np
from docopt import docopt

from fakespikes import util as fsutil

from voltagebudget.neurons import adex
from voltagebudget.neurons import lif

from voltagebudget.util import k_spikes
from voltagebudget.util import estimate_rate
from voltagebudget.util import estimate_communication
from voltagebudget.util import estimate_computation
from voltagebudget.util import mean_budget


def create_simulation(nrn, time, N, ns, ts, time_step=1e-4, **params):
    def simulation(A):
        # Create Y, then Z
        ns_y, ts_y, vs_y = nrn(time,
                               N,
                               ns,
                               ts,
                               A=A,
                               r_b=0,
                               budget=True,
                               report=None,
                               time_step=time_step,
                               **params)

        return ns_y, ts_y, vs_y

    return simulation


if __name__ == "__main__":
    args = docopt(__doc__, version='alpha')
    name = args["NAME"]

    N = int(args["-n"])
    Amax = float(args["-a"])
    n_grid = int(args["--n_grid"])
    f = float(args["-f"])

    w_y = float(args["-w"])
    s_y = float(args["-s"])

    t_stim = float(args["-t"])

    t = 0.3
    if t_stim > 0.2:
        raise ValueError("-t must be less than 0.2 seconds")

    # ---------------------------------------------------------------------
    k = 20
    time_step = 1e-4
    w = 1e-4
    a = 10000

    ns, ts = k_spikes(t_stim, k, w, a=a, dt=time_step, seed=42)
    times = fsutil.create_times(t, time_step)

    # ---------------------------------------------------------------------
    if args["--lif"]:
        nrn = lif
        params = dict(w_in=(w_y, w_y * s_y), bias=(5e-3, 5e-3 / 5), f=f)
    elif args["--adex"]:
        nrn = adex
        params = dict(
            w_in=w_y,
            bias=(5e-10, 5e-10 / 20),
            a=(-1.0e-9, 1.0e-9),
            b=(10e-12, 60.0e-12),
            Ereset=(-48e-3, -55e-3),
            f=f)
    else:
        raise ValueError("opt.py requires neuron type --lif or --adex")

    # ---------------------------------------------------------------------
    # Run
    sim = create_simulation(nrn, t, N, ns, ts, time_step=time_step, **params)
    As = np.linspace(0.0, Amax, n_grid)
    results = [sim(A) for A in As]

    # and unpack results
    ns_ys = [res[0] for res in results]
    ts_ys = [res[1] for res in results]
    vs_ys = [res[2] for res in results]

    # ---------------------------------------------------------------------
    # - Save traces and spikes (in a window)
    # Define larger the window
    times = fsutil.create_times(t, time_step)

    # - Save avg traces, est. firing rates, C, sigma_y, in 1 ms steps
    coincidence_t = 1e-3
    bins = fsutil.create_times(t, coincidence_t)
    budget_names = sorted(vs_ys[0].keys())

    table = []
    for A, vs, ns, ts in zip(As, vs_ys, ns_ys, ts_ys):
        for b in range(1, len(bins) - 1):
            window = [bins[b - 1], bins[b]]

            # Est firing rate
            rate = estimate_rate(ns, ts, window)

            # Est comm
            C = estimate_communication(times, ns, ts, window)

            # Est comp
            sigma_y = estimate_computation(times, ns, ts, window)

            # Avg budgets
            m = mean_budget(times, vs, window)

            # Join all into a row, appending A and bin time 
            t1 = bins[b - 1]
            row = [t1, A, rate, C, sigma_y] + [m[bn] for bn in budget_names]
            table.append(row)

    header = ["t", "As", "rates", "Cs", "sigma_ys"] + budget_names
    with open("{}.csv".format(name), "wb") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(header)
        writer.writerows(table)

    # ---------------------------------------------------------------------
    # - Write params
    args = {
        'N': N,
        'Amax': Amax,
        'n_grid': n_grid,
        'f': f,
        'w_y': w_y,
        's_y': s_y,
        't_stim': t_stim
    }
    keys = sorted(args.keys())
    with open("{}_args.csv".format(name), "wb") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(keys)
        writer.writerow([args[key] for key in keys])
