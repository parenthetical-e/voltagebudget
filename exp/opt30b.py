"""Usage: opt30b.py NAME M 
        [-w W] [-s S] [-a A] [-t T] [-f F] [-n N]

Search {A, phi, sigma_in} and maximizing {Vcomp, Vosc}.

    Arguments:
        NAME    results name
        M       number of opt interations

    Options:
        -h --help               show this screen
        -w W                    average input weight [default: 0.15e-9]
        -s S                    std dev of the input weight [default: 0.5]
        -a A                    maximum oscillation size [default: 1e-3]
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
from voltagebudget.neurons import lif
from voltagebudget.util import k_spikes
from voltagebudget.util import estimate_communication
from voltagebudget.util import estimate_computation
from voltagebudget.util import mean_budget


def create_problem(time, window, ns, ts, f, w_in, bias, time_step=1e-4, N=100):
    def problem(pars):
        A = pars[0]
        phi = pars[1]

        # Create Y, then Z
        ns_y, ts_y, vs_y = lif(time,
                               N,
                               ns,
                               ts,
                               w_in=w_in,
                               bias=bias,
                               f=f,
                               A=A,
                               phi=phi,
                               r_b=0,
                               budget=True,
                               report=None,
                               time_step=time_step)

        # Window for opt analysis
        times = fsutil.create_times(time, time_step)
        vs_m = mean_budget(times, vs_y, window)
        comp = vs_m['comp']
        osc = vs_m['osc']

        print("opt: ({}, {}); par: (A {}, phi {})".format(comp, osc, A, phi))

        return -comp, -osc

    return problem


if __name__ == "__main__":
    args = docopt(__doc__, version='alpha')
    name = args["NAME"]

    M = int(args["M"])
    N = int(args["-n"])
    w_y = float(args["-w"])
    s_y = float(args["-s"])
    f = float(args["-f"])
    Amax = float(args["-a"])

    # ---------------------------------------------------------------------
    # Create input
    t = 0.3

    t_stim = float(args["-t"])
    if t_stim > 0.2:
        raise ValueError("-t must be less than 0.2 seconds")

    k = 20
    time_step = 1e-4
    w = 1e-4
    a = 10000
    ns, ts = k_spikes(t_stim, k, w, a=a, dt=time_step, seed=42)
    times = fsutil.create_times(t, time_step)

    # ---------------------------------------------------------------------
    # Intialize the problem
    w_in = [w_y, w_y * s_y]
    bias = [5e-3, 5e-3 / 5]
    window = [t_stim + 1e-3, t_stim + 4e-3]

    sim = create_problem(
        t, window, ns, ts, f, w_in, bias, N=N, time_step=time_step)

    problem = Problem(2, 2)
    problem.types[:] = [Real(0.0, Amax), Real(0.0, (1 / f) * 0.5)]
    problem.function = sim

    # ---------------------------------------------------------------------
    # Run
    algorithm = NSGAII(problem)
    algorithm.run(M)

    # ---------------------------------------------------------------------
    # Process results
    results = dict(
        v_comp=[s.objectives[0] for s in algorithm.result],
        v_osc=[s.objectives[1] for s in algorithm.result],
        As=[s.variables[0] for s in algorithm.result],
        phis=[s.variables[1] for s in algorithm.result])

    # Simulate params, want sigma_comp and C
    Cs = []
    sigma_ys = []
    l = len(results['As'])

    for i in range(l):
        A = results['As'][i]
        phi = results['phis'][i]

        # Create Y, then Z
        ns_y, ts_y, vs_y = lif(t,
                               N,
                               ns,
                               ts,
                               w_in=w_in,
                               bias=bias,
                               f=f,
                               A=A,
                               phi=phi,
                               r_b=0,
                               budget=True,
                               report=None,
                               time_step=time_step)

        C = estimate_communication(
            times, ns_y, ts_y, window, time_step=time_step)
        sigma_y = estimate_computation(times, ns_y, ts_y, window)

        Cs.append(C)
        sigma_ys.append(sigma_y)

    results['Cs'] = Cs
    results['sigma_ys'] = sigma_ys

    # Write
    keys = sorted(results.keys())
    with open("{}.csv".format(name), "wb") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(keys)
        writer.writerows(zip(* [results[key] for key in keys]))

    # Write args
    args = {'N': N, 'Amax': Amax, 'f': f, 'w_y': w_y, 't_stim': t_stim}
    keys = sorted(args.keys())
    with open("{}_args.csv".format(name), "wb") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(keys)
        writer.writerow([args[key] for key in keys])
