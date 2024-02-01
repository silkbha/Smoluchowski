import numpy as np

def sigma(i,j):
    """ Collisional cross section.
    """
    return

def v_brown(m_i,m_j):
    """ Brownian motion component to be added to relative particle velocities.
        Allows for treatment of collisions between same-sized particles
        (otherwise their relative velocities would be zero -> no collisions -> no coagulation/fragmentation).
    """
    return 0


def C():
    #TODO fix placeholder conditions in if-statements

    # epsilon = (m_n - (m_i+m_j)) / (m_n - m_m)
    # if m_i > m_j:
    #     return epsilon
    # elif m_j < m_i:
    #     return 1-epsilon
    # else:
    #     return 0
    return



def Podolak(dust):
    """
    """
    
    St = dust[0]
    masses = dust[1]
    densities = dust[2] # number density
    velos = dust[3]

    densities_new = np.zeros([len(St)])
    for k,(St_k,m_k,n_k,v_k) in enumerate(zip(St,masses,densities,velos)):

        dndt_gain = 0
        dndt_loss = 0
        for i,(St_i,m_i,n_i,v_i) in enumerate(zip(St,masses,densities,velos)):

            # Define relative velocity with added Browninan motion term.
            vrel_ik = np.abs(v_k-v_i) + v_brown(m_k,m_i)

            # Mass loss due to coagulation
            dndt_loss += n_k*n_i * sigma(St_k,St_i) * vrel_ik
            
            # Mass gain due to coagulation 
            for j,(St_j,m_j,n_j,v_j) in enumerate(zip(St,masses,densities,velos)):
                vrel_ij = np.abs(v_j-v_i) + v_brown(m_j,m_i)
                dndt_gain += n_i*n_j * sigma(St_i,St_j) * vrel_ij * C(i,j,k)
        
        # Get new n_k(t+1) by adding dndt to previous n_k(t).
        # (factor 0.5 in gain term to prevent double counting)
        densities_new[k] = n_k + 0.5 * dndt_gain - dndt_loss
    
    return densities_new