# TODO - max, min, mean Vf: what is impact on C(Z) and lev(Y)
# TODO - review pareto exps, what still makes sense to rerun?
# TODO compare C and lev. What is opt using Vf? Compare to V things.
# TODO - search for global random opt
# TODO - search for precise opt 
# TODO - search for precise opt using I-E interactions. (precise I-E, and highly lateral I)

# %matplotlib inline
# import matplotlib.pyplot as plt
from __future__ import division

import csv

import numpy as np
from docopt import docopt

from fakespikes import neurons, rates
from fakespikes import util as fsutil
from platypus.algorithms import NSGAII
from platypus.core import Problem
from platypus.types import Real
from voltagebudget.neurons import lif
from voltagebudget.util import k_spikes
from voltagebudget.util import estimate_communication
from voltagebudget.util import estimate_computation
from voltagebudget.util import mean_budget
from voltagebudget.util import filter_budget
from voltagebudget.util import filter_spikes
from voltagebudget.util import filter_times
from voltagebudget.learn import lif_brute, coincidence_detection

#!
n_jobs = 12

# Learn to detect CC (see bias, learn w_in)
# in 100 neurons
N = 100

# - Train data
r = 10  # Hz
M = 50  # 50 neurons
t_train = 2  # length

dt = 0.5e-3  # res
if dt < 1e-3:
    raise ValueError("dt too small")

times = fsutil.create_times(t_train, dt)
rates = np.ones_like(times) * r

nrns = neurons.Spikes(M, t_train, dt=dt, seed=seed)
ns, ts = fsutil.to_spiketimes(times, nrns.poisson(rates))

# - Ground truth
k = 5  # number CC to count as a detection event
ts_y = coincidence_detection(ts, a_tol=1e-3, k=k)
print(len(ts_y), ts_y)

# - Qpt
# Generate unique biases spanning the 'viable' range
biases = np.linspace(1e-3, 5e-3, N)

# Grid size
n_iter = 50

# !
params = []
for bs in biases:
    param = lif_brute(
        t_train,
        ns,
        ts,
        ts_y,
        w_range=(.1e-9, 5e-9),
        bias_range=bs,
        num=n_iter,
        n_jobs=n_jobs,
        diagnostic=False)

    params.append(param)

# A.
# Generate test packet and score it
t_test = .155
t_on = .05

k = 5
w = 100e-3
rate = 20

_, ts_test = poisson_impulse(t, t_stim, w, rate, 50, seed=10, dt=dt)

# Set target window, T
# TODO several? How to handle this?

# Find Vc in T(s?)

# Use inequalties to set A
# 1. Vo > min Vc
# 2. Vo > max Vc (but not equal)
# 3. Vo > mean Vc

# Measure Lev, determinsitic C (Z=20) in T

# B.
# Repeat opt using (A, phi), look at Vo, Vc.
# Loss 1. Lev
#      2. C
#      3. Both (pareto front)

# C. 
# Use precise I,E to deliver (A, phi)
# sin -> E - [tau1] -> Pop
#        E - [tau2] -> I -> [tau3] -> pop
# tune (A. phi, tau1, and tau3) - fix tau2?
# Loss 1. Lev
#      2. C
#      3. Both (pareto front)

# D.
# Consider I type heterogeniety. 
# Different I get phase locked....
# to up, down, near zero-cross
# use this topology to repeat C