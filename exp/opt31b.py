"""Usage: opt31b.py NAME M 
        [-w W] [-a A] [-t T] [-f F] [-n N]

Search {A, phi} and maximizing {Vcomp, Vosc}.

 Arguments:
        NAME    results name
        M       number of opt interations

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
from voltagebudget.util import estimate_communication
from voltagebudget.util import estimate_computation
from voltagebudget.util import mean_budget
from voltagebudget.util import filter_budget
from voltagebudget.util import filter_spikes
from voltagebudget.util import filter_times


def create_problem(time,
                   stim_window,
                   delay_window,
                   ns,
                   ts,
                   f,
                   w_in,
                   a,
                   b,
                   E,
                   bias,
                   time_step=1e-4,
                   N=100):
    def problem(pars):
        A = pars[0]
        phi = pars[1]

        # Create Y, then Z
        ns_y, ts_y, vs_y = adex(
            time,
            N,
            ns,
            ts,
            a=a,
            b=b,
            Ereset=E,
            f=f,
            A=A,
            phi=phi,
            r_b=0,
            budget=True,
            report=None)

        # Window for opt analysis
        times = fsutil.create_times(time, time_step)

        # Filter ns, ts, vs for window
        ns_t, ts_y = filter_spikes(ts_y, ns_y, stim_window)
        vs_y = filter_budget(times, vs_y, stim_window)
        times = filter_times(times, stim_window)

        # Est. the mean budget in the delay_window around t*, the first passage
        # in the network (ns, ts).
        try:
            vs_m = mean_budget(times, N, ns_y, ts_y, vs_y, delay_window)
            comp = vs_m['comp']
            osc = vs_m['osc']
        except ValueError:
            comp = 0.0
            osc = 0.0

        print("opt: ({}, {}); par: (A {}, phi {})".format(comp, osc, A, phi))

        return -comp, -osc

    return problem


if __name__ == "__main__":
    args = docopt(__doc__, version='alpha')
    name = args["NAME"]

    M = int(args["M"])
    N = int(args["-n"])
    w_y = float(args["-w"])
    f = float(args["-f"])
    Amax = float(args["-a"])

    # ---------------------------------------------------------------------
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
    w_in = w_y
    bias = [5e-10, 5e-10 / 20]
    a = [-1.0e-9, 1.0e-9]
    b = [10e-12, 60.0e-12]
    E = [-48e-3, -55e-3]
    stim_window = [t_stim, t_stim + 50e-3]
    delay_window = [-1e-3, 0.0]

    sim = create_problem(
        t,
        stim_window,
        delay_window,
        ns,
        ts,
        f,
        w_in,
        a,
        b,
        E,
        bias,
        time_step=time_step,
        N=N)

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
        ns_y, ts_y, vs_y = adex(
            t,
            N,
            ns,
            ts,
            a=a,
            b=b,
            Ereset=E,
            f=f,
            A=A,
            phi=phi,
            r_b=0,
            budget=True,
            report=None)

        C = estimate_communication(
            times, ns_y, ts_y, stim_window, time_step=time_step)
        sigma_y = estimate_computation(times, ns_y, ts_y, stim_window)

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
