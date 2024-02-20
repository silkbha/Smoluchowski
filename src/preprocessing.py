import numpy as np

def E_to_T(E_gas):
    """ Converts kinetic energy of gas to temperature using kinetic theory of gases.
    """
    k_B = 1.380649e-16 # Boltzmann constant in erg/K.
    return 2 * E_gas / (3 * k_B)

def St_to_r(St, rho_gas,T_gas,specvol):
    """ Converts particle Stokes number to absolute particle size.
        From: C.P. Dullemond slides.
        For now only in Epstein drag regime. (TODO add Stokes regime(s) later?)
        Inputs: 
            - St       : Stokes number(s) to be converted. (scalar or array)
            - rho_gas  : gas mass density in g/cm3.
            - T_gas    : gas temperature in K.
            - specvol  : dust monomer specific volume in cm3/g (i.e. 1/rho_dust).
        Outputs:
            - Particle size(s) in cm. (scalar or array)
    """
    # Omega_K = 1 # Keplerian orbital velocity (always 1 in FARGO shearing box?).
    k_B = 1.380649e-16 # Boltzmann constant in erg/K.
    m_p = 1.6726219236951e-24 # Proton mass in g.
    # Calculate thermal velocity of gas in cm/s.
    v_th = np.sqrt( (8 * k_B * T_gas) / (np.pi * 1 * m_p) ) # 1 = mean molecular weight (assumes 100% HI).
    # Calculate particle size.
    return St * rho_gas * v_th * specvol

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


def preprocessing_podolak(dustinfo,duststate,gasstate):
    """ Takes state and info given by FARGO3D and converts to input for dust evolution step.


        # TODO convert 3D velocity distribution to 2D relative velocity matrix
    """


    # Dust Info
    Stokes = dustinfo           # (N,1) array
    Nbins = len(dustinfo)

    # Dust state
    densities = duststate[:,0]  # (N,1) array
    velos = duststate[:,1:]     # (N,3) array

    # Gas state
    rho_gas  = gasstate[0]      # single value
    E_gas    = gasstate[1]      # single value
    c_s      = gasstate[2]      # single value
    rho_dust = gasstate[3]      # single value

    T_gas = E_to_T(E_gas)
    
    sizes  = np.zeros(Nbins)
    masses = np.zeros(Nbins)

    vrel   = np.zeros((Nbins,Nbins))
    sigma  = np.zeros((Nbins,Nbins))



    return densities,vrel,sigma
