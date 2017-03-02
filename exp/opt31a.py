"""Usage: opt31a.py NAME N 
        (--lif | --adex)
        [-w W] [-a A] [-t T] [-f F] [-n N]

Search {A, phi, a, b, Ereset} and maximizing {Vcomp, Vosc}.

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
from voltagebudget.util import estimate_communication
from voltagebudget.util import estimate_computation


def create_problem(nrn, time, t_stim, N, ns, ts, f, Nz=100, **params):
    def problem(pars):
        A = pars[0]
        phi = pars[1]
        a = pars[2]
        b = pars[3]
        E = pars[4]

        # Reset sigma_in
        params["a"][1] = a
        params["b"][1] = b
        params["Ereset"][1] = E

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
            return 0.0, 0.0

        # Window for opt analysis
        t0 = t_stim + 3e-3
        tn = t_stim + 5e-3

        times = fsutil.create_times(t, 1e-4)
        m = np.logical_and(times >= t0, times <= tn)

        comp = vs_y['comp'][:, m].mean()
        osc = vs_y['osc'][:, m].mean()

        print("opt: ({}, {}); par: (A {}, phi {}, a {}, b {}, Ereset {})".
              format(comp, osc, A, phi, a, b, E))

        return -comp, -osc

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
        raise NotImplementedError("--lif not supported; try opt20?")
        # nrn = lif
        # params = dict(w_in=[0.3e-9, 0.3e-9 / 2], bias=[5e-3, 5e-3 / 5])
    elif args["--adex"]:
        nrn = adex
        params = dict(
            w_in=w_y,
            bias=(5e-10, 5e-10 / 20),
            a=[-1.0e-9, 1.0e-9],
            b=[10e-12, 60.0e-12],
            Ereset=[-48e-3, -55e-3])
    else:
        raise ValueError("opt21.py requires neuron type --lif")

    sim = create_problem(nrn, t, t_stim, k, ns, ts, f=f, **params)

    # ---------------------------------------------------------------------
    problem = Problem(5, 2)
    problem.types[:] = [
        Real(0.0, Amax), Real(0.0, (1 / f) * 0.5), Real(-1.0e-9, 1.0e-9),
        Real(10e-12, 60.0e-12), Real(-48e-3, -55e-3)
    ]

    problem.function = sim
    algorithm = NSGAII(problem)
    algorithm.run(N)

    # - Results
    results = dict(
        v_comp=[s.objectives[0] for s in algorithm.result],
        v_osc=[s.objectives[1] for s in algorithm.result],
        As=[s.variables[0] for s in algorithm.result],
        phis=[s.variables[1] for s in algorithm.result],
        a=[s.variables[2] for s in algorithm.result],
        b=[s.variables[3] for s in algorithm.result],
        Ereset=[s.variables[4] for s in algorithm.result])

    # Simulate params, want sigma_comp and C
    Cs = []
    sigma_comps = []
    l = len(results['As'])
    for i in range(l):
        # Reset sigma_in
        params["a"][1] = results['a'][i]
        params["b"][1] = results['b'][i]
        params["Ereset"][1] = results['Ereset'][i]

        A = results['As'][i]
        phi = results['phis'][i]

        # Create Y, then Z
        ns_y, ts_y, vs_y = nrn(t,
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

        C = estimate_communication(t_stim, ns_y, ts_y)
        sigma_comp = estimate_computation(ns_y, ts_y)

        Cs.append(C)
        sigma_comps.append(sigma_comp)

    results['Cs'] = Cs
    results['sigma_comps'] = sigma_comps

    # Write
    keys = sorted(results.keys())
    with open("{}.csv".format(name), "wb") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(keys)
        writer.writerows(zip(* [results[key] for key in keys]))

    # - Write args
    args = {'N': N, 'Amax': Amax, 'f': f, 'w_y': w_y, 't_stim': t_stim}
    keys = sorted(args.keys())
    with open("{}_args.csv".format(name), "wb") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(keys)
        writer.writerow([args[key] for key in keys])
