# %matplotlib inline
# import matplotlib.pyplot as plt
# import numpy as np

from fakespikes import util as fsutil
from platypus.algorithms import NSGAII
from platypus.core import Problem
from platypus.types import Real
from voltagebudget import *

# ---------------------------------------------------------------------
t = 0.3

k = 20
t_stim = 0.1

dt = 1e-4
w = 1e-4
a = 10000
ns, ts = util.k_spikes(t_stim, k, w, a=a, dt=dt, seed=None)
print(len(ts))

times = fsutil.create_times(t, dt)

# ---------------------------------------------------------------------
f = 50
nrn = neurons.lif
sim = neurons.create_problem(nrn, t_stim, k, ns, ts, f)

# ---------------------------------------------------------------------
Amax = 30e-3
problem = Problem(1, 2)
problem.types[:] = Real(0.0, Amax)

problem.function = sim
algorithm = NSGAII(problem)
algorithm.run(1000)

As = [s.objectives[0] for s in algorithm.result]
Cs = [s.objectives[1] for s in algorithm.result]
