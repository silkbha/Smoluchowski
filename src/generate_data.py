import numpy as np
from matplotlib import pyplot as plt
# from preprocessing import generate_inputs_basic as generate_inputs
from preprocessing import get_mass
from preprocessing import MRN

# pick one
from podolak import evolve_simple as evolve
# from podolak import evolve_modified as evolve




# def parameter_sweep():
#     """
#     """
#     nbins   = 100

#     Rho_gas = np.logspace(0.1,10,100)
#     C_s     = np.logspace(0.1,1,100)
    
#     for rho_gas in Rho_gas:
#         for c_s in C_s:
#             for idxmin0 in range(nbins):
#                 for idxmax0 in range(nbins):
#                     sizes,masses,densities,velos,T_gas = generate_inputs(nbins,idxmin0,idxmax0,rho_gas,c_s)
#                     densities_new = evolve(sizes,masses, densities,velos,T_gas)
#     #TODO save, track, test
    
#     return

def test(nbins,steps):
    """ TODO track and check total mass conservation
        TODO save, track, plot density evolution
    """
    idxmin0 = 1
    idxmax0 = 3
    
    rho_gas = 1e-12
    c_s     = 1e2

    densities = np.zeros((steps,nbins))
    # sizes,masses, densities[0], velos, T_gas = generate_inputs(nbins,idxmin0,idxmax0,rho_gas,c_s)
    sizes  = np.logspace(10**(0.1e-2),10**(0.5e-2),nbins)
    masses = get_mass(sizes, rho_dust=1e-1)
    velos  = np.zeros((len(masses),3))
    T_gas  = 204

    densities[0] = MRN(sizes, sizes[idxmin0],sizes[idxmax0], rho_gas)

    for k in range(steps-1):
        densities[k+1] = evolve(sizes,masses, densities[k],velos,T_gas)
    
    return densities



if __name__=="__main__":

    nbins   = 20
    steps   = 5

    densities = test(nbins,steps)
    colors = plt.cm.jet(np.linspace(0,1,steps))
    
    for i in range(steps):
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        ax.plot(np.arange(20),densities[i], color=colors[i])
        plt.show()
    