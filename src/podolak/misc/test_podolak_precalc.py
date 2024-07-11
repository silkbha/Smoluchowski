import numpy as np

def St_to_r(St,Omega_K, rho_gas,T_gas,rho_dust):
    """ Converts particle Stokes number to absolute particle size.
        From: C.P. Dullemond slides.
        For now only in Epstein regime. (TODO add Stokes regime later?)
    """
    # Calculate thermal velocity of gas in cm/s.
    k_B = 1.380649e-16 # Boltzmann constant in erg/K
    m_p = 1.6726219236951e-24 # Proton mass in g
    v_th = np.sqrt( (8 * k_B * T_gas) / (np.pi * 1 * m_p) ) # 1 = mean molecular weight (assumes 100% HI).
    
    # Calculate particle size.
    return St * rho_gas * v_th / (Omega_K * rho_dust)
Stokes_to_size = np.vectorize(St_to_r)

def vrel_bm(m_i,m_j,T_gas):
    """ Brownian motion component to be added to relative particle velocities.
        Source: Birnstiel et al. 2010 (A&A 513, A79), Eq. 44.
        Allows for treatment of collisions between same-sized particles, i.e. m_i == m_j.
        Otherwise, their relative velocities would always be zero -> no collisions -> no coagulation/fragmentation.
    """
    k_B = 1.380649e-16 # Boltzmann constant in erg/K
    return np.sqrt( (8 * (m_i+m_j) * k_B * T_gas) / (np.pi*m_i*m_j) )
vrels_BM = np.vectorize(vrel_bm)

def cross(r_i,r_j):
    """ Collisional cross section.
    """
    return np.pi * (r_i+r_j)**2



def find_idx_low(array,target):
    """ Finds the index of the nearest value below a given target value inside a given numpy array.
        Source: https://stackoverflow.com/questions/67617053/
    """
    diff = target - array
    diff[diff < 0] = np.inf
    idx = diff.argmin()
    return idx

def C(masses,i,j,k):
    """ 
    """
    
    m_s = masses[i] + masses[j]
    
    # Nearest bins for which m_m < m_s < m_n
    m = find_idx_low(masses, m_s)
    n = m + 1
    
    if k != m and k != n:
        return 0
    elif m == k == len(masses)-1:
        # Edge case: index n is out of bounds!
        return 0 # TODO what to do here s.t. mass is still conserved (e.g. final bin as sink particle?)

    m_m = masses[m]
    m_n = masses[n]
    epsilon = (m_n - m_s) / (m_n - m_m)

    if k == m: # i.e. if m_k = m_m
        return epsilon
    elif k == n:
        return 1 - epsilon
    else:
        raise ValueError("Something went wrong when computing nearest neighboring mass bins!")



#################
# Main Function #
#################
def podolak(dustinfo,duststate,gasstate):
    """ Single-step time evolution of the Smoluchowski coagulation equation in 0D using the Podolak algorithm.
        Based on Brauer et al. 2008 (A&A 480, 859-877), Appendix A.1.
    
        Input:
            - dustinfo  = !SORTED! 2D array containing time-invariant information on dust size bins.
                - col 1/2 : Stokes number corresponding to each bin
                - col 2/2 : particle mass corresponding to each bin (easier to store than to compute)
            - duststate = 2D array containing current state information of dust at time t0.
                - col 1/3 : particle number density per size bin
                - col 2/3 : Keplerian orbital frequency per size bin
                - col 3/3 : particle velocity in x direction per size bin
                - col 4/5 : particle velocity in y direction per size bin
                - col 5/5 : particle velocity in z direction per size bin
            - gasstate  = 1D array containing current state information of gas at time t0.
                - gas mass density
                - gas temperature
                - dust particle mass density
        Output:
            - array of number density distribution at time t0 + dt
                - col 1/1 : particle number density per size bin

        TODO 3D velocities!
        TODO Omega_K given by Fargo or need to calculate manually here?
        TODO St to r conversion now only in Epstein... add Stokes regime(s) later?
    """
    
    # Check if input dust info & state arrays are given correctly.
    if len(dustinfo[0]) != len(duststate[0]):
        raise ValueError("Dust info and state arrays must contain the same number of bins!")

    ########################################################################################
    # Dust info
    Stokes = dustinfo[:,0]      # Nx1 array
    masses = dustinfo[:,1]      # Nx1 array
    N_bins = len(Stokes)

    # Dust state
    densities = duststate[:,0]  # Nx1 array
    Omega_K = duststate[:,1]    # Nx1 array
    velos = duststate[:,2:]     # Nx3 array

    # Gas state
    rho_gas = gasstate[0]       # single value
    T_gas = gasstate[1]         # single value
    rho_dust = gasstate[2]      # single value
    ########################################################################################

    # Create array of real (absolute) particle sizes.
    sizes = Stokes_to_size(Stokes,Omega_K, rho_gas,T_gas,rho_dust)

    # Create matrix of relative particle velocities
    vrel = np.zeros((N_bins,N_bins))

    # Create matrix of particle cross sections
    sigma = np.zeros((N_bins,N_bins))

    # Calculate evolved size distribution.
    densities_new = np.zeros(len(densities))
    for k,n_k in enumerate(densities):

        dndt_gain = 0
        dndt_loss = 0

        # TODO Over what range i & j? To avoid edge cases...
        for i,n_i in enumerate(densities):

            # Mass loss due to coagulation
            dndt_loss += n_k*n_i * sigma[i,k] * vrel[i,k]
            
            # Mass gain due to coagulation 
            for j,n_j in enumerate(densities):
                C_ijk = C(masses,i,j,k)
                if C_ijk == 0:
                    continue # dndt_gain += 0
                dndt_gain += n_i*n_j * sigma[i,j] * vrel[i,j] * C_ijk
        
        # Get new n_k(t+1) by adding dndt to previous n_k(t).
        # (with factor 0.5 in gain term to prevent double counting of collisions ij & ji)
        densities_new[k] = n_k + 0.5 * dndt_gain - dndt_loss
    
    return densities_new


if __name__=="__main__":
    print("Hello There")