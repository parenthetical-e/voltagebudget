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
from voltagebudget.util import write_spikes

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


def create_stim(name,
                t,
                stim_number=40,
                stim_onset=0.2,
                stim_offset=0.250,
                stim_rate=8,
                seed_stim=7525,
                time_step=1e-5,
                verbose=False):
    if verbose:
        print(">>> Building input.")

    ns, ts = poisson_impulse(
        t,
        stim_onset,
        stim_offset - stim_onset,
        stim_rate,
        dt=time_step,
        n=stim_number,
        seed=seed_stim)

    if verbose:
        print(">>> {} spikes generated.".format(ns.size))
        print(">>> Saving input.")

    write_spikes("{}.csv".format(name), ns, ts)
