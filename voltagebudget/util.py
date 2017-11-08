from __future__ import division
import csv
import os
import json
import voltagebudget
import numpy as np

from scipy.signal import square
from fakespikes import neurons, rates
from fakespikes import util as fsutil


def step_waves(I, f, duty, t, dt):
    times = fsutil.create_times(t, dt)

    wave = I * square(2 * np.pi * f * times - np.pi, duty=duty)
    wave[wave < 0] = 0.0

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
    w_max = initial_inputs['w_max']
    bias = initial_inputs['bias']
    sigma = initial_inputs['sigma']

    return params, w_max, bias, sigma


def get_mode_names(json_path=None):
    """List all modes"""
    if json_path is None:
        json_path = os.path.join(
            os.path.split(voltagebudget.__file__)[0], 'modes.json')

    with open(json_path, 'r') as data_file:
        modes = json.load(data_file)

    return modes.keys()


def get_default_modes():
    """Get the default modes built into voltagebudget library"""
    json_path = os.path.join(
        os.path.split(voltagebudget.__file__)[0], 'modes.json')

    with open(json_path, 'r') as data_file:
        modes = json.load(data_file)

    return modes


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


def estimate_communication(times,
                           ns,
                           ts,
                           window,
                           coincidence_t=1e-3,
                           coincidence_n=20,
                           return_all=False,
                           time_step=1e-4):

    # Define overall analysis window 
    t0 = window[0]
    tn = window[1]
    if tn + coincidence_t > times.max():
        raise ValueError("Final window must be less than max value in times")

    m = np.logical_and(t0 <= ts, ts <= tn)
    ts = ts[m]
    ns = ns[m]

    # Calculate C for every possible coincidence (CC) window, for all time
    Cs = []
    for t in times:
        # Get CC window
        cc0 = t
        ccn = t + coincidence_t
        m = np.logical_and(cc0 <= ts, ts <= ccn)

        # Count spikes in the window
        C_t = 0
        if ts[m].size > 0:
            n_spikes = ts[m].size
            C_t = max(n_spikes - coincidence_n, 0) / coincidence_n

        Cs.append(C_t)

    # Find avg C
    C = np.max(Cs)
    out = C

    if return_all:
        out = (C, Cs)

    return out


def precision(ns, ts, ns_ref, ts_ref, combine=True):
    """Analyze spike time precision (jitter)
    
    Parameters
    ----------
    ns : array-list (1d)
        Neuron codes 
    ts : array-list (1d, seconds)
        Spikes times 
    ns_ref : array-list (1d)
        Neuron codes for the reference train
    ts_ref : array-list (1d, seconds)
        Spikes times for the reference train
    """

    prec = []
    ns_prec = []

    # Join all ns, into the '0' key?
    if combine:
        ns = np.zeros_like(ns)
        ns_ref = np.zeros_like(ns_ref)

    # isolate units, and reformat
    ref = fsutil.to_spikedict(ns_ref, ts_ref)
    target = fsutil.to_spikedict(ns, ts)

    # analyze precision
    for n, r in ref.items():
        try:
            x = target[n]
        except KeyError:
            x = np.zeros_like(r)

        minl = min(len(r), len(x))
        diffs = np.abs([r[i] - x[i] for i in range(minl)])

        prec.append(np.mean(diffs))
        ns_prec.append(n)

    # If were are combining return scalars
    # not sequences
    if combine:
        prec = prec[0]
        ns_prec = ns_prec[0]

    return ns_prec, prec
