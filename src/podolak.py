import numpy as np
from numpy import heaviside as Theta

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



def find_m(array,target):
    """ Finds the index of the nearest value below a given target value in an array.
        Based on: https://stackoverflow.com/questions/67617053/ .
    """
    diff = target - array
    diff[diff < 0] = np.inf
    idx = diff.argmin()
    return idx

def C(masses, i,j,k):
    """ Calculates coefficient C_ijk for the Podolak coagulation algorithm.
        From: Brauer et al. 2008 (A&A 480, 859-877), Equation A.5.
    """
    m_s = masses[i] + masses[j]
    
    # Nearest bins m & n for which m_m < m_s < m_n
    m = find_m(masses, m_s)
    n = m + 1
    
    if k != m and k != n:
        return 0
    elif n == len(masses):
        # Edge case: m is final index, index n is out of bounds.
        return m_s / masses[k] # final bin acts as sink particle
    
    m_m = masses[m]
    m_n = masses[n]
    epsilon = (m_n - m_s) / (m_n - m_m)

    if k == m: # i.e. if m_k = m_m
        return epsilon
    elif k == n:
        return 1 - epsilon
    else:
        raise ValueError("Something went wrong when computing coefficient C...")

def D(masses, j,k, c_e):
    """ Calculates coefficient D_jk for the modified Podolak coagulation algorithm.
        From: Brauer et al. 2008 (A&A 480, 859-877), Equation A.9.
        TODO: edge cases
    """
    m_j  = masses[j]
    m_k  = masses[k]

    if j <= k+1-c_e:
        if k+1 == len(masses):
            # Edge case: k is final index, k+1 is out of bounds.
            # TODO
            raise ValueError("Reached edge case: index out of bounds. Will fix later...")
        m_k1 = masses[k+1]
        return -m_j / (m_k1 - m_k)
    elif j > k+1-c_e:
        return -1
    else:
        raise ValueError("Something went wrong when computing coefficient D...")

def E(masses, j,k, c_e):
    """ Calculates coefficient E_jk for the modified Podolak coagulation algorithm.
        From: Brauer et al. 2008 (A&A 480, 859-877), Equation A.10.
        TODO: edge cases
        TODO: correct theta?
    """
    m_j  = masses[j]
    m_k  = masses[k]
    m_k0 = masses[k-1]

    if j <= k-c_e:
        return m_j / (m_k - m_k0)
    elif j > k-c_e:
        if k+1 == len(masses):
            # Edge case: k is final index, k+1 is out of bounds.
            # TODO
            raise ValueError("Reached edge case: index out of bounds. Will fix later...")
        m_k1 = masses[k+1]
        return (1 - (m_j + m_k0 - m_k)/(m_k1 - m_k)) * Theta(m_k1 - m_j - m_k0, 0)
    else:
        raise ValueError("Something went wrong when computing coefficient E...")

def find_ce(masses,k):
    """ TODO
    """
    m_k  = masses[k]
    m_k0 = masses[k-1]
    dmk = m_k - m_k0

    return

def M(masses, i,j,k):
    """ Calculates coefficient M_ijk for the modified Podolak coagulation algorithm.
        From: Brauer et al. 2008 (A&A 480, 859-877), Equation A.13.
        TODO: c_e
        TODO: edge cases
    """
    c_e = find_ce(masses,k)

    M = C(masses,i,j,k) * Theta(k-i-1.5, 0) * Theta(i-j-0.5, 0)
    
    # Kronecker deltas
    if i == k:
        M += D(masses,j,i,c_e)
    elif i == k-1:
        M += E(masses,j,i+1,c_e) * Theta(k-j-1.5)
    if i == j:
        M += 0.5 * C(masses,i,j,k)
    
    return M



###############################################################################################################
#                                             Main Functions                                                  #
###############################################################################################################

def evolve_simple(sizes,masses, densities,velos,T_gas):
    """ Single-step time evolution of the Smoluchowski coagulation equation in 0D using the Podolak algorithm.
        Based on: Brauer et al. 2008 (A&A 480, 859-877), Appendix A.1.
    
        Input:
            ========
            - sizes
            - masses
            ========
            - densities
            - velos
            - T_gas
        Output:
            - particle number density per size bin at time t0 + dt.
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
                
                C_ijk = C(masses,i,j,k)
                if C_ijk == 0:
                    continue # dndt_gain += 0

                vrel_ij = np.linalg.norm(v_j-v_i) + vrel_bm(m_j,m_i,T_gas)
                dndt_gain += n_i*n_j * sigma(r_i,r_j) * vrel_ij * C_ijk
        
        # Get new n_k(t+1) by adding dndt to previous n_k(t).
        # (with factor 0.5 in gain term to prevent double counting of collisions ij & ji)
        densities_new[k] = n_k + 0.5 * dndt_gain - dndt_loss
    
    return densities_new


def evolve_modified(sizes,masses, densities,velos,T_gas):
    """ Single-step time evolution of the Smoluchowski coagulation equation in 0D using the Podolak algorithm.
        Modified version to accomodate larger value ranges with better numerical precision for mass conservation.
        Based on: Brauer et al. 2008 (A&A 480, 859-877), Appendix A.2.
    
        Input:
            ========
            - sizes
            - masses
            ========
            - densities
            - velos
            - T_gas
        Output:
            - particle number density per size bin at time t0 + dt.
    """
    
    # Check if input dust info & state arrays are given correctly.
    # TODO

    # Calculate evolved size distribution.
    densities_new = np.zeros(len(densities))
    for k,n_k in enumerate(densities):

        dndt = 0

        for i,(r_i,m_i,n_i,v_i) in enumerate(zip(sizes,masses,densities,velos)):
            for j,(r_j,m_j,n_j,v_j) in enumerate(zip(sizes,masses,densities,velos)):
                
                M_ijk = M(masses,i,j,k)
                if M_ijk == 0:
                    continue # dndt += 0

                vrel_ij = np.linalg.norm(v_j-v_i) + vrel_bm(m_j,m_i,T_gas)
                dndt += n_i*n_j * sigma(r_i,r_j) * vrel_ij * M_ijk

        # Get new n_k(t+1) by adding dndt to previous n_k(t).
        densities_new[k] = n_k + dndt
    
    return densities_new

if __name__=="__main__":
    print("Hello There")