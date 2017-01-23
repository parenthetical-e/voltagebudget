"""Usage: opt2.py NAME N 
        (--lif | --adex)
        [-a A] [-w W] [-b B]

Minimize A and maximisze C.

    Arguments:
        NAME    results name (.hdf5)
        N       number of interations

    Options:
        -h --help               show this screen
        -a A                    maximum oscillation size [default: 30e-3]
        -w W                    maximum `w_in` size [default: 0.2e-9]
        -b B                    maximum `bias` size [default: 5e-3]
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


def create_problem(nrn, t_stim, N, ns, ts, f, pad=10e-3, Nz=100):
    time = np.max(ts) + pad

    def problem(vars):
        A = vars[0]
        w_in = vars[1]
        bias = vars[2]

        # Create Y, then Z
        ns_y, ts_y, _ = nrn(time,
                            N,
                            ns,
                            ts,
                            w_in=w_in,
                            bias=bias,
                            f=f,
                            A=A,
                            r_b=0.0,
                            sigma_scale=0.0,
                            report=None)

        # If Y didn't spike, C=0
        if ns_y.shape[0] == 0:
            return A, 0.0

        _, ts_z, _ = lif(time,
                         Nz,
                         ns_y,
                         ts_y,
                         w_in=0.1e-9,
                         bias=10e-6,
                         r_b=0,
                         sigma_scale=10,
                         f=0,
                         A=0,
                         report=None)

        # Est communication
        m = np.logical_or(t_stim <= ts_z, ts_z <= (t_stim + pad))
        C = 0
        if ts_z[m].size > 0:
            C = ts_z[m].size / float(Nz)

        # Return losses (with C in mimization form)
        print(A, -C)
        return A, -C

    return problem


if __name__ == "__main__":
    args = docopt(__doc__, version='alpha')
    name = args["NAME"]
    N = int(args["N"])

    Amax = float(args["-a"])
    win_max = float(args["-w"])
    bias_max = float(args["-b"])

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
    f = 50
    if args["--lif"]:
        sim = create_problem(lif, t_stim, k, ns, ts, f)
    elif args["--adex"]:
        sim = create_problem(adex, t_stim, k, ns, ts, f)
    else:
        raise ValueError("opt2.py requires neuron type --lif or --adex")

    # ---------------------------------------------------------------------
    problem = Problem(3, 2)
    problem.types[:] = [Real(0.0, Amax), Real(0.0, win_max), Real(0.0,
                                                                  bias_max)]

    problem.function = sim
    algorithm = NSGAII(problem)
    algorithm.run(N)

    results = dict(
        As=[s.objectives[0] for s in algorithm.result],
        Cs=[s.objectives[1] for s in algorithm.result])

    keys = sorted(results.keys())
    with open("{}.csv".format(name), "wb") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(keys)
        writer.writerows(zip(*[results[key] for key in keys]))
