import numpy as np

a  = np.linspace(0.1,10,50) # np.array([0.1, 0.5, 1., 5., 10.]) #
S0 = np.logspace(-4,3,200)
t = np.linspace(1e-9,1e3,2000)

par_space = len(a)*len(S0)*len(t)
print("Param space size:", len(a)*len(S0)*len(t))