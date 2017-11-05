from __future__ import division
import csv
import os
import json
import voltagebudget
import numpy as np

from scipy.signal import square
from copy import deepcopy
from fakespikes import neurons, rates
from fakespikes import util as fsutil


def step_waves(I, f, duty, t, dt):
    times = fsutil.create_times(t, dt)

    wave = I * square(2 * np.pi * f * times - np.pi, duty=duty)

    return wave


def poisson_impulse(t, t_stim, w, rate, n=10, dt=1e-3, seed=None):
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
            # Is a number?
            try:
                v = float(v)
            except ValueError:
                pass

            # Add or init?
            if k in data:
                data[k].append(v)
            else:
                data[k] = [
                    v,
                ]

    return data


def read_modes(mode, json_path=None):
    # Read in modes from the detault location
    # or with what the user provided?
    if json_path is None:
        json_path = os.path.join(
            os.path.split(voltagebudget.__file__)[0], 'modes.json')

    with open(json_path, 'r') as data_file:
        modes = json.load(data_file)

    # Extract params
    params = modes[mode]

    # And default input
    initial_inputs = params.pop('initial_inputs')
    w_in = initial_inputs['w_in']
    bias = initial_inputs['bias']

    return params, w_in, bias


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
        if k == 'seed':
            v = int(v)
        else:
            try:
                v = float(v)
            except ValueError:
                pass

        # Convert bools
        if v.strip() == 'True':
            v = True
        if v.strip() == 'False':
            v = False

        # save
        args_data[k] = v

    return args_data
