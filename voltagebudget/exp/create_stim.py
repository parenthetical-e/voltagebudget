import csv
import os
import numpy as np

import voltagebudget
from voltagebudget.util import poisson_impulse
from voltagebudget.util import write_spikes


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

    write_spikes("{}".format(name), ns, ts)
