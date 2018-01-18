import numpy as np
from voltagebudget.util import mad


def optimal_coordination(ts, p, dt):
    if dt < 0:
        raise ValueError("dt must be positive.")
    if not (0 <= p <= 1):
        raise ValueError("p must be between (0-1).")

    # Est initial and target
    initial = mad(ts)
    target = initial - (initial * p)

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


def reverse():
    # TODO
    pass
