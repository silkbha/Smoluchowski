import numpy as np

def vrel_bm(m_i,m_j,T_gas):
    """ Brownian motion component to be added to relative particle velocities.
        From: Birnstiel et al. 2010 (A&A 513, A79), Eq. 44.
        Allows for treatment of collisions between same-sized particles, i.e. m_i == m_j.
        Otherwise, their relative velocities would always be zero -> no collisions -> no coagulation/fragmentation.
        Inputs :
            - m_i,m_j : particle mass in g.
            - T_gas   : gas temperature in K.
        Output :
            - Brownian motion induced relative particle velocity in cm/s.
    """
    k_B = 1.380649e-16 # Boltzmann constant in erg/K
    return np.sqrt( (8 * (m_i+m_j) * k_B * T_gas) / (np.pi*m_i*m_j) )

def sigma(r_i,r_j):
    """ Collisional cross section.
        Inputs:
            - r_i,r_j : particle size in cm.
        Output:
            - Collisional cross section in cm2.
    """
    return np.pi * (r_i+r_j)**2



def find_idx_low(array,target):
    """ Finds the index of the nearest value below a given target value in an array.
        Based on: https://stackoverflow.com/questions/67617053/ .
    """
    diff = target - array
    diff[diff < 0] = np.inf
    idx = diff.argmin()
    return idx

def C(masses,i,j,k):
    """ Calculates coefficient C_ijk for the Podolak coagulation algorithm.
        From: Brauer et al. 2008 (A&A 480, 859-877), Equation A.5.
    """
    m_s = masses[i] + masses[j]
    
    # Nearest bins m & n for which m_m < m_s < m_n
    m = find_idx_low(masses, m_s)
    n = m + 1
    
    if k != m and k != n:
        return 0
    elif n == len(masses):
        # Edge case: m is final index, index n is out of bounds!
        return m_s / masses[k] # final bin acts as sink particle
    
    m_m = masses[m]
    m_n = masses[n]
    epsilon = (m_n - m_s) / (m_n - m_m)

    if k == m: # i.e. if m_k = m_m
        return epsilon
    elif k == n:
        return 1 - epsilon
    else:
        raise ValueError("Something went wrong when computing nearest neighboring mass bins!")



###############################################################################################################
#                                              Main Function                                                  #
###############################################################################################################

def evolve(sizes,masses,densities,velos,T_gas):
    """ Single-step time evolution of the Smoluchowski coagulation equation in 0D using the Podolak algorithm.
        Based on: Brauer et al. 2008 (A&A 480, 859-877), Appendix A.1.
    
        Input:
            - dustinfo  = Sorted (N,1) array containing time-invariant information on dust size bins.
                - col 1/1 : Stokes number corresponding to each bin
                - col 2/2 : particle mass corresponding to each bin (easier to store than to compute)
            - duststate = (N,4) array containing current state information of dust at time t0.
                - col 1/4 : particle number density per size bin
                - col 2/4 : particle velocity in x direction per size bin
                - col 3/4 : particle velocity in y direction per size bin
                - col 4/4 : particle velocity in z direction per size bin
            - gasstate  = List containing current state information of gas at time t0.
                - gas mass density
                - gas temperature
                - dust monomer mass density
        Output:
            - array of number density distribution at time t0 + dt
                - col 1/1 : particle number density per size bin
    """
    
    # Check if input dust info & state arrays are given correctly.
    # TODO

    # Calculate evolved size distribution.
    densities_new = np.zeros(len(densities))
    for k,(r_k,m_k,n_k,v_k) in enumerate(zip(sizes,masses,densities,velos)):

        dndt_gain = 0
        dndt_loss = 0

        for i,(r_i,m_i,n_i,v_i) in enumerate(zip(sizes,masses,densities,velos)):

            # Define relative velocity with added Browninan motion term.
            vrel_ik = np.linalg.norm(v_k-v_i) + vrel_bm(m_k,m_i,T_gas)

            # Mass loss due to coagulation
            dndt_loss += n_k*n_i * sigma(r_k,r_i) * vrel_ik
            
            # Mass gain due to coagulation 
            for j,(r_j,m_j,n_j,v_j) in enumerate(zip(sizes,masses,densities,velos)):
                vrel_ij = np.linalg.norm(v_j-v_i) + vrel_bm(m_j,m_i,T_gas)
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