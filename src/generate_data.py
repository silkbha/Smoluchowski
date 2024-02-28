import numpy as np
from podolak import evolve_simple as evolve
from preprocessing import preprocessing_direct as prep


def MRN(a, amin,amax,rho_gas, r=3.5):
    """
    """
    da = amax**(4-r) - amin**(4-r)
    return (4-r) * a**(-r) / (4/3 * np.pi * da * rho_gas)


def generate_inputs():

    # Stays constant throughout parameter sweep
    # TODO keep constant and omit from input space altogether?
    Stokes = np.logspace(0.01,0.5,100)


    Masses = np.zeros(100) # compute from sizes and rho_dust? Or given by Fargo?

    # Variable parameters (ranges)
    # TODO number density distributions: how to choose appropriate parameter ranges????
    # TODO velos: input each 3D particle velocity or compute relative velo matrix first?

    # TODO gas parameters: single values, easy to set ranges

    # vary over MRN distribution!! minmax, slope
    # Loop over variable param grid!



    # First try: zero vrel, brownian only : check analytical solution dullemond/dominik 2005
    return

def parameter_sweep():
    return


if __name__=="__main__":
    generate_inputs()