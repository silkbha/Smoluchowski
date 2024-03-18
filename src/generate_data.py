import numpy as np
from podolak import evolve_simple as evolve
from preprocessing import preprocessing_direct as prep


def MRN(a, amin,amax,rho_gas, r=3.5):
    """
    """
    da = amax**(4-r) - amin**(4-r)
    return (4-r) * a**(-r) / (4/3 * np.pi * da * rho_gas)


def generate_inputs(nbins,idxmin0,idxmax0 ,rho_gas,c_s):
    """ # First try: zero vrel, brownian only : check analytical solution dullemond/dominik 2005.
    """
    rho_dust = 1e1
    St_min, St_max = 0.01, 0.5
    Stokes = np.logspace(St_min,St_max,nbins)

    sizes, masses, T_gas = prep(Stokes,rho_dust,rho_gas,c_s)
    densities = MRN(sizes, sizes[idxmin0],sizes[idxmax0], rho_gas)
    velos = np.zeros((len(Stokes),3))

    return sizes, masses, densities, velos, T_gas

def parameter_sweep():

    nbins   = 100

    Rho_gas = np.logspace(0.1,10,100)
    C_s     = np.logspace(0.1,1,100)
    
    for rho_gas in Rho_gas:
        for c_s in C_s:
            for idxmin0 in range(nbins):
                for idxmax0 in range(nbins):
                    sizes,masses,densities,velos,T_gas = generate_inputs(nbins,idxmin0,idxmax0,rho_gas,c_s)
                    densities_new = evolve(sizes,masses, densities,velos,T_gas)
    #TODO save, track, etc.
    
    return


if __name__=="__main__":
    parameter_sweep()