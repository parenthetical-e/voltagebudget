"""Usage: opt3.py NAME N 
        (--lif | --adex)
        [-w W]

Minimize w_in and maximize C. 

Noisy bias enforces uniqeness.

    Arguments:
        NAME    results name (.hdf5)
        N       number of interations

    Options:
        -h --help               show this screen
        -w W                    maximum `w_in` size [default: 0.2e-9]
"""

# %matplotlib inline
# import matplotlib.pyplot as plt
from __future__ import division

import csv

import numpy as np
from docopt import docopt

from fakespikes import util as fsutil
from platypus.algorithms import NSGAII
from platypus.core import Problem
from platypus.types import Real
from voltagebudget.neurons import adex, lif
from voltagebudget.util import k_spikes


def create_problem(nrn, t_stim, N, ns, ts, pad=10e-3, Nz=100, **params):
    time = np.max(ts) + pad

    def problem(vars):
        w_in = vars[0]

        # Create Y, then Z
        ns_y, ts_y = nrn(time,
                         N,
                         ns,
                         ts,
                         w_in=w_in,
                         f=0.0,
                         A=0.0,
                         r_b=0.0,
                         report=None,
                         budget=False,
                         **params)

        # If Y didn't spike, C=0
        if ns_y.shape[0] == 0:
            print("Null Y.")
            return w_in, 0.0

        _, ts_z = lif(time,
                      Nz,
                      ns_y,
                      ts_y,
                      w_in=(0.1e-9, 0.1e-9 / 10),
                      bias=(10e-6, 10e-6 / 10),
                      r_b=0,
                      f=0,
                      A=0,
                      refractory=t_stim + pad,
                      budget=False,
                      report=None)

        # Est communication
        m = np.logical_or(t_stim <= ts_z, ts_z <= (t_stim + pad))
        C = 0
        if ts_z[m].size > 0:
            C = ts_z[m].size / float(Nz)

        # Return losses (with C in mimization form)
        print(w_in, -C)
        return w_in, -C

    return problem


if __name__ == "__main__":
    args = docopt(__doc__, version='alpha')
    name = args["NAME"]
    N = int(args["N"])

    win_max = float(args["-w"])

    # ---------------------------------------------------------------------
    t = 0.3

    k = 20
    t_stim = 0.1

    dt = 1e-4
    w = 1e-4
    a = 10000
    ns, ts = k_spikes(t_stim, k, w, a=a, dt=dt, seed=42)
    times = fsutil.create_times(t, dt)

    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    if args["--lif"]:
        nrn = lif
        params = dict(bias=(5e-3, 5e-3 / 10))
    elif args["--adex"]:
        nrn = adex
        params = dict(
            bias=(5e-10, 5e-10 / 20),
            a=(-1.0e-9, 1.0e-9),
            b=(10e-12, 60.0e-12),
            Ereset=(-48e-3, -55e-3))
    else:
        raise ValueError("opt.py requires neuron type --lif or --adex")

    sim = create_problem(nrn, t_stim, k, ns, ts, **params)

    # ---------------------------------------------------------------------
    problem = Problem(1, 2)
    problem.types[:] = [Real(0.0, win_max)]

    problem.function = sim
    algorithm = NSGAII(problem)
    algorithm.run(N)

    results = dict(
        Cs=[s.objectives[1] for s in algorithm.result],
        ws=[s.objectives[0] for s in algorithm.result])

    keys = sorted(results.keys())
    with open("{}.csv".format(name), "wb") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(keys)
        writer.writerows(zip(*[results[key] for key in keys]))
