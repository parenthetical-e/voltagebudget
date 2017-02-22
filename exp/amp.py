"""Usage: amp.py NAME 
        (--lif | --adex)
        [-w W] [-a A] [-f F] [-n N] [-t T]
        [--n_grid NGRID]

Explore oscillation amplitude's effect on communication and computation.

    Arguments:
        NAME    results name (.hdf5)

    Options:
        -h --help               show this screen
        -w W                    average input weight [default: 0.2e-9]
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
from voltagebudget.neurons import adex, lif
from voltagebudget.util import k_spikes


def create_simulation(nrn,
                      time,
                      t_stim,
                      N,
                      ns,
                      ts,
                      pad=20e-3,
                      Nz=100,
                      **params):
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
                               **params)

        # If Y didn't spike, C=0
        if ns_y.shape[0] == 0:
            print("Null Y.")
            return 0.0, 0.0, None

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

        # Est comp
        times = fsutil.create_times(t, 1e-4)
        comp = vs_y['comp']

        m = np.logical_and(times >= t0, times <= tn)
        sigma_comp = comp[:, m].std()

        # Est communication
        m = np.logical_or(t0 <= ts_z, ts_z <= tn)
        C = 0
        if ts_z[m].size > 0:
            C = ts_z[m].size / float(Nz)

        return C, sigma_comp, ns_y, ts_y, vs_y

    return simulation


if __name__ == "__main__":
    args = docopt(__doc__, version='alpha')
    name = args["NAME"]

    N = int(args["-n"])
    Amax = float(args["-a"])
    n_grid = int(args["--n_grid"])
    f = float(args["-f"])

    w_y = float(args["-w"])
    t_stim = float(args["-t"])

    t = 0.3
    if t_stim > 0.2:
        raise ValueError("-t must be less than 0.2 seconds")

    # ---------------------------------------------------------------------
    k = 20
    dt = 1e-4
    w = 1e-4
    a = 10000

    ns, ts = k_spikes(t_stim, k, w, a=a, dt=dt, seed=42)
    times = fsutil.create_times(t, dt)

    # ---------------------------------------------------------------------
    if args["--lif"]:
        nrn = lif
        params = dict(w_in=(w_y, w_y / 2), bias=(5e-3, 5e-3 / 5), f=f)
    elif args["--adex"]:
        nrn = adex
        params = dict(
            w_in=w_y,  # 0.6e-9
            bias=(5e-10, 5e-10 / 20),
            a=(-1.0e-9, 1.0e-9),
            b=(10e-12, 60.0e-12),
            Ereset=(-48e-3, -55e-3),
            f=f)
    else:
        raise ValueError("opt.py requires neuron type --lif or --adex")

    sim = create_simulation(nrn, t, t_stim, k, ns, ts, **params)

    As = np.linspace(0.0, Amax, n_grid)
    results = [sim(A) for A in As]

    # - C, sigma_y, ns_y, ts_y, vs_y
    Cs = [res[0] for res in results]
    sigma_comp = [res[1] for res in results]
    ns_ys = [res[2] for res in results]
    ts_ys = [res[3] for res in results]
    vs_ys = [res[4] for res in results]

    results = dict(As=As, Cs=Cs, sigma_comp=sigma_comp)

    # ---------------------------------------------------------------------
    # - Save traces and spikes (in a window)

    # Define the window
    t0 = t_stim - 10e-3
    tn = t_stim + 50e-3

    # Save spikes
    At = []
    for _ in ns_ys:
        a = np.repeat(A, _.shape[0]).tolist()
        At.append(a)
    ns_ys = np.concatenate(ns_ys)
    ts_ys = np.concatenate(ts_ys)
    At = np.concatenate(At)

    m = np.logical_or(t0 <= ts_ys, ts_ys <= tn)

    np.savetxt(
        '{}_spks.csv'.format(name),
        np.vstack([At[m], ns_ys[m], ts_ys[m]]).T,
        delimiter=',',
        header="A,n,t")

    # - Save traces 
    dt_sim = 1e-4
    times = fsutil.create_times(t, dt_sim)

    m = np.logical_and(times >= t0, times <= tn)

    free = []
    osc = []
    comp = []
    vm = []
    At = []
    timest = []
    for A, v in zip(As, vs_ys):
        fr = v["free"][:, m]
        o = v["osc"][:, m]
        c = v["comp"][:, m]
        mm = v["vm"][:, m]
        a = np.repeat(A, fr.shape[1]).tolist()

        free.append(fr)
        osc.append(o)
        comp.append(c)
        vm.append(mm)
        At.extend(a)

        timest.extend(times[m])

    free = np.hstack(free)
    free = np.vstack([At, timest, free])
    np.savetxt(
        '{}_free.csv'.format(name),
        free.T,
        delimiter=",",
        header="A,time," + ",".join([str(i) for i in range(N)]))

    osc = np.hstack(osc)
    osc = np.vstack([At, timest, osc])
    np.savetxt(
        '{}_osc.csv'.format(name),
        osc.T,
        delimiter=",",
        header="A,time," + ",".join([str(i) for i in range(N)]))

    comp = np.hstack(comp)
    comp = np.vstack([At, timest, comp])
    np.savetxt(
        '{}_comp.csv'.format(name),
        comp.T,
        delimiter=",",
        header="A,time," + ",".join([str(i) for i in range(N)]))

    vm = np.hstack(vm)
    vm = np.vstack([At, timest, vm])
    np.savetxt(
        '{}_vm.csv'.format(name),
        vm.T,
        delimiter=",",
        header="A,time," + ",".join([str(i) for i in range(N)]))

    # - Write results
    keys = sorted(results.keys())
    with open("{}.csv".format(name), "wb") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(keys)
        writer.writerows(zip(* [results[key] for key in keys]))

    # - Write params
    args = {
        'N': N,
        'Amax': Amax,
        'n_grid': n_grid,
        'f': f,
        'w_y': w_y,
        't_stim': t_stim
    }
    keys = sorted(args.keys())
    with open("{}_args.csv".format(name), "wb") as fi:
        writer = csv.writer(fi, delimiter=",")
        writer.writerow(keys)
        writer.writerow([args[key] for key in keys])
