from __future__ import division
import csv

from fakespikes import neurons, rates
from fakespikes import util as fsutil

import numpy as np
from copy import deepcopy


def poisson_impulse(t, t_stim, w, rate, n, dt=1e-3, seed=None):
    """Create a pulse of spikes w seconds wide, starting at t_stim."""

    # Poisson sample the rate over w
    times = fsutil.create_times(t, dt)
    nrns = neurons.Spikes(n, t, dt=dt, seed=seed)
    pulse = rates.square_pulse(times, rate, t_stim, w, dt, min_a=0)

    ns, ts = fsutil.to_spiketimes(times, nrns.poisson(pulse))

    return ns, ts


def filter_spikes(ns, ts, window):
    m = np.logical_and(ts >= window[0], ts <= window[1])

    return ns[m], ts[m]


def filter_budget(times, vs, window):
    m = np.logical_and(times >= window[0], times <= window[1])

    filtered = {}
    for k, v in vs.items():
        try:
            len(vs[k])  # error on scalar/float
            filtered[k] = vs[k][:, m]
        except IndexError:
            filtered[k] = vs[k][m]
        except TypeError:
            filtered[k] = vs[k]  # copy over scalar values

    return filtered


def _read_csv_cols_into_dict(filename):
    # csv goes here:
    data = {}

    # Open and iterate over lines
    # when csv is returning lines as dicts
    # using the header as a key
    reader = csv.DictReader(open(filename, 'r'))
    for row in reader:
        for k, v in row.items():
            # Add or init?
            if k in data:
                data[k].append(v)
            else:
                data[k] = [
                    v,
                ]

    return data


def read_stim(stim):
    """Read in budget_experiment stimulation, as a dict"""
    return _read_csv_cols_into_dict(stim)


def read_results(results):
    """Read in budget_experiment results, as a dict"""
    return _read_csv_cols_into_dict(results)


def read_args(args):
    """Read in an adex arguments file, as a dict"""
    reader = csv.reader(open(args, 'r'))
    args_data = {}
    for row in reader:
        k = row[0]
        v = row[1]

        # Convert numbers
        try:
            v = float(v)
        except ValueError:
            pass

        # Convert bools
        if v == ' True':
            v = True
        if v == 'False':
            v = False

        # save
        args_data[k] = v

    return args_data