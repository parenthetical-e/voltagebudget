import csv
import os
# import voltagebudget
import numpy as np

from fakespikes import neurons, rates
from fakespikes import util as fsutil


def filter_spikes(ns, ts, window):
    m = np.logical_and(ts >= window[0], ts <= window[1])

    return ns[m], ts[m]


def select_n(n, ns, ts):
    m = n == ns
    return ns[m], ts[m]


def locate_first(ns, ts, combine=False):
    # Recode neurons as coming from one neuron,
    # i.e. a hack to examine the network
    if combine:
        ns = np.zeros_like(ns)

    ns_first, ts_first = [], []
    for n in np.unique(ns):
        ns_n, ts_n = select_n(n, ns, ts)

        loc = np.argsort(ts_n).argmin()

        ns_first.append(ns_n[loc])
        ts_first.append(ts_n[loc])

    return np.asarray(ns_first), np.asarray(ts_first)


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
