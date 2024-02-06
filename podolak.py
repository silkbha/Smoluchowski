import numpy as np

def vrel_bm(m_i,m_j,T_gas):
    """ Brownian motion component to be added to relative particle velocities.
        Source: Birnstiel et al. 2010 (A&A 513, A79), Eq. 44.
        Allows for treatment of collisions between same-sized particles, i.e. m_i == m_j.
        Otherwise, their relative velocities would always be zero -> no collisions -> no coagulation/fragmentation.
    """
    k_B = 1.380649e-16 # Boltzmann constant in CGS units (erg/K)
    # k_B = 1.380649e-23 # Boltzmann constant in SI units (J/K)
    return np.sqrt( (8 * (m_i+m_j) * k_B * T_gas) / (np.pi*m_i*m_j) )

def sigma_manual(r_i,r_j):
    """ Collisional cross section.
    """
    return np.pi * (r_i+r_j)**2

def C(m_i,m_j,m_k):
    
    # TODO nearest-neighbor binning for m & n
    m_n = 0
    m_m = 0

    epsilon = (m_n - (m_i+m_j)) / (m_n - m_m)

    # TODO fix placeholder conditions in if-statements
    if True:
        return epsilon
    elif False:
        return 1-epsilon
    else:
        return 0


#################
# Main Function #
#################
def podolak(dustinfo,duststate,gasstate,sigma=None):
    """ Single-step time evolution of the Smoluchowski coagulation equation at one point in space using the Podolak algorithm.
        Source: Brauer et al. 2008 (A&A 480, 859-877), Appendix A.1.
    
        Input:
            - dustinfo  = 2D array containing time-invariant information on dust size bins.
                - row 1/2 : Stokes number corresponding to each bin
                - row 2/2 : particle mass corresponding to each bin (easier to store than to compute)
            - duststate = 2D array containing current state information of dust at time t0.
                - row 1/2 : particle number density per size bin
                - row 2/2 : particle velocity per size bin
            - gasstate  = 1D array containing current state information of gas at time t0.
                - velocity
                - temperature
                - density
            - sigma     = 2D array containing collisional cross sections for each dust size bin combination.
                          This matrix is symmetric (sigma[i,j] == sigma[j,i]).
                          If empty or none, calculate manually from dustinfo.
        Output:
            - array of number density distribution at time t0 + dt
                - row 1/1 : particle number density per size bin
    """
    
    # Check if input dust info & state arrays are correctly given.
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

    # Check sigma array (collisional cross-sections).
    if sigma == None or not sigma.any():
        raise ValueError("Array sigma must exist and must not be empty.")
        # TODO
        # If no sigma array given, calculate manually.

    # Calculate evolved size distribution.
    densities_new = np.zeros([len(densities)])
    for k,(St_k,m_k,n_k,v_k) in enumerate(zip(St,masses,densities,velos)):

        dndt_gain = 0
        dndt_loss = 0
        for i,(St_i,m_i,n_i,v_i) in enumerate(zip(St,masses,densities,velos)):

            # Define relative velocity with added Browninan motion term.
            vrel_ik = np.abs(v_k-v_i) + vrel_bm(m_k,m_i,T_gas)

            # Mass loss due to coagulation
            dndt_loss += n_k*n_i * sigma[k,i] * vrel_ik
            
            # Mass gain due to coagulation 
            for j,(St_j,m_j,n_j,v_j) in enumerate(zip(St,masses,densities,velos)):
                vrel_ij = np.abs(v_j-v_i) + vrel_bm(m_j,m_i)
                dndt_gain += n_i*n_j * sigma[i,j] * vrel_ij * C(m_i,m_j,m_k)
        
        # Get new n_k(t+1) by adding dndt to previous n_k(t).
        # (factor 0.5 in gain term to prevent double counting)
        densities_new[k] = n_k + 0.5 * dndt_gain - dndt_loss
    
    return densities_new


if __name__=="__main__":
    print("Hello")