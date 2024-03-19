import numpy as np
from podolak import evolve_simple as evolve
from preprocessing import generate_inputs_basic as generate_inputs

def parameter_sweep():
    """
    """
    nbins   = 100

    Rho_gas = np.logspace(0.1,10,100)
    C_s     = np.logspace(0.1,1,100)
    
    for rho_gas in Rho_gas:
        for c_s in C_s:
            for idxmin0 in range(nbins):
                for idxmax0 in range(nbins):
                    sizes,masses,densities,velos,T_gas = generate_inputs(nbins,idxmin0,idxmax0,rho_gas,c_s)
                    densities_new = evolve(sizes,masses, densities,velos,T_gas)
    #TODO save, track, test
    
    return

def test():
    """ TODO track and check total mass conservation
        TODO save, track, plot density evolution
    """
    nbins   = 100
    steps   = 100

    idxmin0 = 1
    idxmax0 = 10
    
    rho_gas = 1e-6
    c_s     = 1e1

    densities = np.zeroes((nbins,steps))
    sizes,masses, densities[:,0], velos, T_gas = generate_inputs(nbins,idxmin0,idxmax0,rho_gas,c_s)
    for k in range(steps-1):
        densities[:,k+1] = evolve(sizes,masses, densities[:,k],velos,T_gas)

    return

if __name__=="__main__":
    test()