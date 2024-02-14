import numpy as np

def St_to_r(St,rho_gas,T_gas,rho_dust,Omega_K):
    """ Converts particle Stokes number to absolute particle size.
        Source: C.P. Dullemond slides.
        For now only in Epstein regime. (TODO add Stokes regime later?)
    """
    # Calculate thermal velocity of gas in cm/s.
    k_B = 1.380649e-16 # Boltzmann constant in erg/K
    m_p = 1.6726219236951e-24 # Proton mass in g
    v_th = np.sqrt( (8 * k_B * T_gas) / (np.pi * 1 * m_p) ) # 1 = mean molecular weight (assumes 100% HI).
    
    # Calculate particle size.
    return St * rho_gas * v_th / (Omega_K * rho_dust)

def vrel_bm(m_i,m_j,T_gas):
    """ Brownian motion component to be added to relative particle velocities.
        Source: Birnstiel et al. 2010 (A&A 513, A79), Eq. 44.
        Allows for treatment of collisions between same-sized particles, i.e. m_i == m_j.
        Otherwise, their relative velocities would always be zero -> no collisions -> no coagulation/fragmentation.
    """
    k_B = 1.380649e-16 # Boltzmann constant in erg/K
    return np.sqrt( (8 * (m_i+m_j) * k_B * T_gas) / (np.pi*m_i*m_j) )

def sigma(r_i,r_j):
    """ Collisional cross section.
    """
    return np.pi * (r_i+r_j)**2



def find_idx_low(array,target):
    """ Finds the index of the nearest value below a given target value inside a given numpy array.
        Source: https://stackoverflow.com/questions/67617053/find-nearest-value-above-and-below-a-value-inside-numpy-array
    """
    diff = target - array
    diff[diff < 0] = np.inf
    idx = diff.argmin()
    return idx

def C(masses,i,j,k):
    """ TODO edge case: what happens when k is last index? Index n would then be out of bounds...
    """
    
    m_s = masses[i] + masses[j]
    
    # Nearest bins for which m_m < m_s < m_n
    m = find_idx_low(masses, m_s)
    n = m + 1
    
    if k != m and k != n:
        return 0

    m_m = masses[m]
    m_n = masses[n] # TODO if m=k and k is last index, then n out of bounds!
    epsilon = (m_n - m_s) / (m_n - m_m)

    if k == m: # i.e. if m_k = m_m
        return epsilon
    elif k == n:
        return 1 - epsilon
    else:
        raise ValueError("Error computing nearest neighboring mass bins!")



#################
# Main Function #
#################
def podolak(dustinfo,duststate,gasstate):
    """ Single-step time evolution of the Smoluchowski coagulation equation in 0D using the Podolak algorithm.
        Source: Brauer et al. 2008 (A&A 480, 859-877), Appendix A.1.
    
        Input:
            - dustinfo  = !SORTED! 2D array containing time-invariant information on dust size bins.
                - row 1/2 : Stokes number corresponding to each bin
                - row 2/2 : particle mass corresponding to each bin (easier to store than to compute)
            - duststate = 2D array containing current state information of dust at time t0.
                - row 1/3 : particle number density per size bin
                - row 2/3 : particle velocity per size bin
                - row 3/3 : Keplerian orbital frequency per size bin
            - gasstate  = 1D array containing current state information of gas at time t0.
                - gas mass density
                - gas velocity
                - gas temperature
                - dust particle mass density
        Output:
            - array of number density distribution at time t0 + dt
                - row 1/1 : particle number density per size bin
    """
    
    # Check if input dust info & state arrays are correctly given.
    if dustinfo.shape != duststate.shape:
        raise ValueError("Dust info and state arrays must contain the same number of bins!")

    ########################################################################################
    # Dust info
    Stokes = dustinfo[0]
    masses = dustinfo[1]

    # Dust state
    densities = duststate[0]
    velos = duststate[1]
    Omega_K = duststate[2]
    
    # Gas state
    rho_gas = gasstate[0]       # single value
    v_gas = gasstate[1]         # single value
    T_gas = gasstate[2]         # single value
    rho_dust = gasstate[3]      # single value
    ########################################################################################

    # Create array of real particle sizes.
    sizes = np.zeros(len(Stokes))
    for x,(St,w_K) in enumerate(zip(Stokes,Omega_K)):
        sizes[x] = St_to_r(St,rho_gas,T_gas,rho_dust,w_K)

    # Calculate evolved size distribution.
    densities_new = np.zeros(len(densities))
    for k,(r_k,m_k,n_k,v_k) in enumerate(zip(sizes,masses,densities,velos)):

        dndt_gain = 0
        dndt_loss = 0

        # TODO Over what range i & j? To avoid edge cases...
        for i,(r_i,m_i,n_i,v_i) in enumerate(zip(sizes,masses,densities,velos)):

            # Define relative velocity with added Browninan motion term.
            # TODO velocities in 3 dimensions.
            vrel_ik = np.abs(v_k-v_i) + vrel_bm(m_k,m_i,T_gas)

            # Mass loss due to coagulation
            dndt_loss += n_k*n_i * sigma(r_k,r_i) * vrel_ik
            
            # Mass gain due to coagulation 
            for j,(r_j,m_j,n_j,v_j) in enumerate(zip(sizes,masses,densities,velos)):
                vrel_ij = np.abs(v_j-v_i) + vrel_bm(m_j,m_i)
                C_ijk = C(masses,i,j,k)
                if C_ijk == 0:
                    continue # dndt_gain += 0
                dndt_gain += n_i*n_j * sigma(r_i,r_j) * vrel_ij * C_ijk
        
        # Get new n_k(t+1) by adding dndt to previous n_k(t).
        # (with factor 0.5 in gain term to prevent double counting of collisions ij & ji)
        densities_new[k] = n_k + 0.5 * dndt_gain - dndt_loss
    
    return densities_new


if __name__=="__main__":
    print("Hello There")