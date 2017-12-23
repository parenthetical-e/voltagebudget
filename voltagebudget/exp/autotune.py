import json
import csv
import os
import numpy as np

import voltagebudget
from voltagebudget.neurons import adex
from voltagebudget.neurons import shadow_adex
from voltagebudget.util import poisson_impulse
from voltagebudget.util import read_results
from voltagebudget.util import read_stim
from voltagebudget.util import read_args
from voltagebudget.util import read_modes

from voltagebudget.budget import filter_voltages
from voltagebudget.budget import locate_firsts
from voltagebudget.budget import locate_peaks
from voltagebudget.budget import estimate_communication
from voltagebudget.budget import precision

from voltagebudget.exp import forward
from voltagebudget.exp import forward_shadow
from voltagebudget.exp import sweep_power
from voltagebudget.exp import replay
from voltagebudget.exp import reverse
from voltagebudget.exp import create_stim


def autotune_w(mode,
               w_0,
               rate,
               t=3,
               k=20,
               stim_rate=30,
               seed_stim=1,
               max_mult=2):
    # Load cell params
    params, _, bias, sigma = read_modes(mode)

    # Create frozen input spikes
    stim_onset = 0.1
    stim_offset = t
    dt = 1e-5
    ns, ts = poisson_impulse(
        t,
        stim_onset,
        stim_offset - stim_onset,
        stim_rate,
        n=k,
        dt=dt,
        seed=seed_stim)

    # -
    def problem(p):
        w = p[0]

        ns_y, ts_y = adex(
            1,
            t,
            ns,
            ts,
            w_max=w,
            bias=bias,
            sigma=sigma,
            report=None,
            budget=False,
            **params)

        rate_y = ts_y.size / (stim_offset - stim_onset)

        return rate_y - rate

    p0 = [w_0]
    sol = least_squares(problem, p0, bounds=(0, w_0 * max_mult))

    return sol


def autotune_membrane(mode, bias_0, sigma_0, mean, std, t=1):
    # Load cell params
    params, _, _, _ = read_modes(mode)

    # No input spikes
    ns = np.zeros(1)
    ts = np.zeros(1)
    w_max = 0

    # -
    def problem(p):
        bias = p[0]
        sigma = p[0]

        vm, _ = shadow_adex(
            1, t, ns, ts, w_max=w_max, bias=bias, report=None, **params)

        return (np.mean(vm) - mean), (np.std(vm) - std)

    # !
    p0 = [bias_0, sigma_0]
    sol = least_squares(problem, p0)

    return sol
