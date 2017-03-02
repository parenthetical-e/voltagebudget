"""Usage: opt20c.py NAME N 
        (--lif | --adex)
        [-w W] [-a A] [-t T] [-f F] [-n N]

Search {phi, sigma_in} and maximizing {C, sigma_y}.

    Arguments:
        NAME    results name
        N       number of opt interations

    Options:
        -h --help               show this screen
        -w W                    average input weight [default: 0.3e-9]
        -a A                    maximum oscillation size [default: 30e-3]
        -t T                    stim onset time (< 0.2) [default: 0.1]
        -f F                    oscillation frequency (Hz) [default: 50]
        -n N                    number of Y neurons [default: 100]
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


def create_problem(nrn, time, t_stim, N, ns, ts, f, A, Nz=100, **params):
    def problem(pars):
        phi = pars[0]
        sigma_in = pars[1]

        # Reset sigma_in
        params["w_in"][1] = params["w_in"][0] * sigma_in

        # Create Y, then Z
        ns_y, ts_y, vs_y = nrn(time,
                               N,
                               ns,
                               ts,
                               f=f,
                               A=A,
                               phi=phi,
                               r_b=0,
                               budget=True,
                               report=None,
                               **params)

        # If Y didn't spike, C=0
        if ns_y.shape[0] == 0:
            print("Null Y.")
            return np.inf, -0.0

        _, ts_z = lif(time,
                      Nz,
                      ns_y,
                      ts_y,
                      w_in=(0.2e-9, 0.2e-9),
                      bias=(5e-3, 5e-3 / 5),
                      r_b=0,
                      f=0,
                      A=0,
                      refractory=time,
                      budget=False,
                      report=None)

        # Window for opt analysis
        t0 = t_stim + 2e-3
        tn = t_stim + 50e-3

        # Est sigma_comp (variance of the comp)
        times = fsutil.create_times(t, 1e-4)
        comp = vs_y['comp']

        m = np.logical_and(times >= t0, times <= tn)
        sigma_comp = comp[:, m].std()

        # Est communication
        m = np.logical_or(t0 <= ts_z, ts_z <= tn)
        C = 0
        if ts_z[m].size > 0:
            C = ts_z[m].size / float(Nz)

        # Return losses (in mimization form)
        print(sigma_comp, C)
        return -sigma_comp, -C

    return problem


if __name__ == "__main__":
    args = docopt(__doc__, version='alpha')
    name = args["NAME"]

    N = int(args["N"])
    w_y = float(args["-w"])

    f = float(args["-f"])
    Amax = float(args["-a"])

    # ---------------------------------------------------------------------
    t = 0.3

    t_stim = float(args["-t"])
    if t_stim > 0.2:
        raise ValueError("-t must be less than 0.2 seconds")

    k = 20
    dt = 1e-4
    w = 1e-4
    a = 10000
    ns, ts = k_spikes(t_stim, k, w, a=a, dt=dt, seed=42)
    times = fsutil.create_times(t, dt)

    # ---------------------------------------------------------------------
    if args["--lif"]:
        nrn = lif
        params = dict(w_in=[w_y, w_y / 2], bias=[5e-3, 5e-3 / 5])
    elif args["--adex"]:
        raise NotImplementedError("--adex not supported; try opt21?")
    #     nrn = adex
    #     params = dict(
    #         w_in=w_y,
    #         bias=(5e-10, 5e-10 / 20),
    #         a=(-1.0e-9, 1.0e-9),
    #         b=(10e-12, 60.0e-12),
    #         Ereset=(-48e-3, -55e-3))
    else:
        raise ValueError("opt.py requires neuron type --lif")

    sim = create_problem(nrn, t, t_stim, k, ns, ts, f, Amax / 2.0, **params)

    # ---------------------------------------------------------------------
    problem = Problem(3, 2)
    problem.types[:] = [Real(0.0, (1 / f) * 0.5), Real(0.0, 1)]

    problem.function = sim
    algorithm = NSGAII(problem)
    algorithm.run(N)

    # - Results
    results = dict(
        sigma_comp=[s.objectives[0] for s in algorithm.result],
        Cs=[s.objectives[1] for s in algorithm.result],
        phis=[s.variables[1] for s in algorithm.result],
        sigma_in=[s.variables[2] for s in algorithm.result])

    keys = sorted(results.keys())
    with open("{}.csv".format(name), "wb") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(keys)
        writer.writerows(zip(* [results[key] for key in keys]))

    # - Write args
    args = {
        'N': N,
        'Amax': Amax,
        'f': f,
        'w_y': w_y,
        't_stim': t_stim,
        'A': Amax / 2.0
    }
    keys = sorted(args.keys())
    with open("{}_args.csv".format(name), "wb") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(keys)
        writer.writerow([args[key] for key in keys])
