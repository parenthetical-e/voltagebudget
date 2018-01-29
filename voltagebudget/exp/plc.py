import os
import json
import csv
import numpy as np

import voltagebudget
from voltagebudget.util import mad

def even_shift(N, ts, p_var):
    if not (0 <= p_var <= 1):
        raise ValueError("p must be between (0-1).")

    # Calc the total DT needed to achieve p_var

    # Divide by N
    
    # Shift each ts 'inward' toward the mean by DT

    return ts_opt

def max_dev(ts, p_var, dt):
    if dt < 0:
        raise ValueError("dt must be positive.")
    if not (0 <= p_var <= 1):
        raise ValueError("p must be between (0-1).")

    # Est initial and target
    initial = mad(ts)
    target = initial - (initial * p_var)

    # We'll always want the biggest....
    deltas = np.absolute(ts - np.mean(ts))

    # Opt!
    ts_opt = ts.copy()

    while adjusted > target:
        # Find largest delta 
        k = np.argmax(deltas)

        # and shift that spike toward the mean
        if ts_opt[k] < 0:
            ts_opt[k] += dt
        else ts_opt[k] > 0:
            ts_opt[k] -= dt
        else:
            pass

        # Recalc the deltas
        deltas = np.absolute(ts_opt - np.mean(ts_opt))

        # Update rolling MAD
        adjusted = mad(ts_opt)

    return ts_opt
