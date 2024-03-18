import numpy as np
from podolak import evolve_simple as evolve
from preprocessing import preprocessing_direct as prep


def MRN(a, amin,amax,rho_gas, r=3.5):
    """
    """
    da = amax**(4-r) - amin**(4-r)
    return (4-r) * a**(-r) / (4/3 * np.pi * da * rho_gas)


def generate_inputs(St_min,St_max, idxmin0,idxmax0, rho_dust,rho_gas,c_s):

    # Stays constant throughout parameter sweep
    Stokes = np.logspace(St_min,St_max,100)
    sizes,masses,T_gas = prep(Stokes,rho_dust,rho_gas,c_s)

    densities = MRN(sizes, sizes[idxmin0],sizes[idxmax0], rho_gas)
    # Variable parameters (ranges)
    # TODO number density distributions: MRN
    # TODO velos: input each 3D particle velocity or compute relative velo matrix first?

    # TODO gas parameters: single values, easy to set ranges

    # vary over MRN distribution!! minmax, slope
    # Loop over variable param grid!



    # First try: zero vrel, brownian only : check analytical solution dullemond/dominik 2005
    return sizes, masses, densities, T_gas

def parameter_sweep():
    return


if __name__=="__main__":
    generate_inputs()