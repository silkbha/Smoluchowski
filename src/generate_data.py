import numpy as np
from matplotlib import pyplot as plt
from preprocessing import generate_inputs_basic as generate_inputs
from preprocessing import r_to_m
from preprocessing import MRN

# Pick one
from podolak import evolve_simple as evolve
# from podolak import evolve_modified as evolve




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

def test(nbins,steps):
    """ TODO track and check total mass conservation
        TODO save, track, plot density evolution
    """
    # nbins    = 100
    imin     = 1
    imax     = 2
    rho_dust = 1e1

    sizes     = np.logspace(-3,-1,nbins)
    # sizes     = np.linspace(1e-5,5e-5,nbins)
    masses    = r_to_m(sizes,rho_dust)
    
    velos  = np.zeros((len(masses),3))
    T_gas  = 204

    densities = np.zeros((steps,nbins))
    densities[0] = MRN(sizes, imin,imax,rho_dust)
    # sizes,masses, densities[0], velos, T_gas = generate_inputs(nbins,idxmin0,idxmax0,rho_gas,c_s)

    for t in range(steps-1):
        densities[t+1] = evolve(sizes,masses, densities[t],velos,T_gas)
    
    return densities, masses, sizes



if __name__=="__main__":

    nbins   = 100
    steps   = 2

    densities,masses,sizes = test(nbins,steps)
    colors = plt.cm.viridis(np.linspace(0,1,steps))
    
    fig, ax = plt.subplots(1,1, figsize=(7,5))
    for i in range(steps):
        y = densities[i] * masses * sizes
        # fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.plot(sizes,y, color=colors[i])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"a [cm]")
        ax.set_ylabel(r"m $\cdot$ a $\cdot$ f(a) [g cm$^{-3}$]")
    plt.show()
    
    print("Done. Goodbye...")