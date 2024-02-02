import numpy as np

def vrel_brown(m_i,m_j,T_gas):
    """ Brownian motion component to be added to relative particle velocities.
        Allows for treatment of collisions between same-sized particles, i.e. m_i == m_j.
        (otherwise their relative velocities would be zero -> no collisions -> no coagulation/fragmentation)
    """
    k_B = 1.380649e-16 # Boltzmann constant in CGS units (erg/K)
    # k_B = 1.380649e-23 # Boltzmann constant in SI units (J/K)
    return np.sqrt( (8 * (m_i+m_j) * k_B * T_gas) / (np.pi*m_i*m_j) )

def sigma(i,j):
    """ Collisional cross section.
    """
    return 1

def C(m_i,m_j,m_k):
    
    #TODO nearest-neighbor binning for m & n
    m_n = 0
    m_m = 0

    epsilon = (m_n - (m_i+m_j)) / (m_n - m_m)

    #TODO fix placeholder conditions in if-statements
    if True:
        return epsilon
    elif False:
        return 1-epsilon
    else:
        return 0



def podolak(dustinfo,duststate,gasstate):
    """ Single-step time evolution of the Smoluchowski coagulation equation using the Podolak algorithm (Brauer et al. 2008).
    
        Input:
            - dustinfo  = 2D array containing time-invariant information on dust size bins.
                - row 1/2 : Stokes number corresponding to each bin
                - row 2/2 : particle mass corresponding to each bin (easier to store than compute)
            - duststate = 2D array containing current state information of dust at time t0.
                - row 1/2 : particle number density per size bin
                - row 2/2 : particle velocity per size bin
            - gasstate = 1D array containing current state information of gas at time t0.
                - velocity
                - temperature
                - density
        Output:
            - array of number density distribution at time t0 + dt
                - row 1/1 : particle number density per size bin
    """
    
    if dustinfo.shape != duststate.shape:
        raise ValueError("Dust info and state arrays must contain the same number of bins.")

    ########################################################################################
    # Dust info
    St = dustinfo[0]
    masses = dustinfo[1]

    # Dust state
    densities = duststate[0] # number density
    velos = duststate[1]
    
    # Gas state
    v_gas = gasstate[0]
    T_gas = gasstate[1]
    n_gas = gasstate[2]
    ########################################################################################

    densities_new = np.zeros([len(St)])
    for k,(St_k,m_k,n_k,v_k) in enumerate(zip(St,masses,densities,velos)):

        dndt_gain = 0
        dndt_loss = 0
        for i,(St_i,m_i,n_i,v_i) in enumerate(zip(St,masses,densities,velos)):

            # Define relative velocity with added Browninan motion term.
            vrel_ik = np.abs(v_k-v_i) + vrel_brown(m_k,m_i,T_gas)

            # Mass loss due to coagulation
            dndt_loss += n_k*n_i * sigma(St_k,St_i) * vrel_ik
            
            # Mass gain due to coagulation 
            for j,(St_j,m_j,n_j,v_j) in enumerate(zip(St,masses,densities,velos)):
                vrel_ij = np.abs(v_j-v_i) + vrel_brown(m_j,m_i)
                dndt_gain += n_i*n_j * sigma(St_i,St_j) * vrel_ij * C(m_i,m_j,m_k)
        
        # Get new n_k(t+1) by adding dndt to previous n_k(t).
        # (factor 0.5 in gain term to prevent double counting)
        densities_new[k] = n_k + 0.5 * dndt_gain - dndt_loss
    
    return densities_new


if __name__=="__main__":
    print("Hello")