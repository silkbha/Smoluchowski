import os
import sys
from pathlib import Path

import numpy as np

# import dustpy as dp
from dustpy import Simulation
from dustpy import constants as c

# dustpy_kernels.py in main repo directory, import from there:
src_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = str(Path(src_dir).parents[0])
sys.path.append(main_dir)
print(main_dir)
from dustpy_kernels import *

# CONSTANT KERNEL

# create simulation
sim = Simulation()

# Change ini params and initialize simulation

sim.ini.dust.allowDriftingParticles = False
sim.ini.grid.Nr = 3
sim.ini.grid.Nmbpd = 28

sim.initialize()

# Run test coagulation simulation with constant kernel

a = 1.
S0 = 1.

set_constant_kernel(sim, a, S0)