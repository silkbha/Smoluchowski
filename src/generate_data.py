import numpy as np
from podolak import evolve



def main():

    # Stays constant throughout parameter sweep
    # TODO keep constant and omit from input space altogether?
    Stokes = np.logspace(0.01,0.5,100)
    Masses = np.zeros(100) # compute from sizes and rho_dust? Or given by Fargo?

    # Variable parameters (ranges)
    # TODO number density distributions: how to choose appropriate parameter ranges????
    # TODO velos: input each 3D particle velocity or compute relative velo matrix first?

    # TODO gas parameters: single values, easy to set ranges


    # Loop over variable param grid!


    return



if __name__=="__main__":
    main()