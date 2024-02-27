import numpy as np

def cs_to_T(c_s, k_B,m_p,mu):
    """ Converts sound speed of gas (as given by FARGO) to temperature.
        Uses the following relation: p = c_s^2 * rho = kT*rho/mu*mp
    """
    return c_s**2 / (mu * m_p * k_B)

def St_to_r(St, rho_gas,T_gas,rho_dust, k_B,m_p,mu):
    """ Converts particle Stokes number to absolute particle size.
        From: C.P. Dullemond slides.
        Omega_K = 1 # Keplerian orbital velocity (always 1 in FARGO shearing box).
        For now only in Epstein drag regime. (TODO add Stokes regime(s) later?)
        Inputs: 
            - St       : Stokes number(s) to be converted. (scalar or array)
            - rho_gas  : gas mass density in g/cm3.
            - T_gas    : gas temperature in K.
            - rho_dust : dust monomer mass density in g/cm3.
        Outputs:
            - Particle size(s) in cm. (scalar or array)
    """
    # Calculate thermal velocity of gas in cm/s.
    v_th = np.sqrt( (8 * k_B * T_gas) / (np.pi * mu * m_p) )
    # Calculate particle size.
    return St * rho_gas * v_th / rho_dust

def get_mass(r, rho):
    """ Get particle mass from size and density.
        Assumes spherical shape for all particles.
    """
    return rho * 4/3 * np.pi * r**3



def preprocessing_direct(Stokes,rho_dust,rho_gas,c_s):
    """ Takes state and info given by FARGO3D and converts to input for dust evolution step.

        Input:
            - Stokes number per size bin
            - dust monomer mass density
            - gas mass density
            - gas sound speed
        Output:
            - sizes
            - masses
            - T_gas

        # TODO convert 3D velocity distribution to 2D relative velocity matrix?
    """

    # Constants
    k_B = 1.380649e-16          # Boltzmann constant in erg/K.
    m_p = 1.6726219236951e-24   # Proton mass in g.
    mu  = 2.33                  # Mean molecular weight (molecular Hydrogen)

    T_gas  = cs_to_T(c_s, k_B,m_p,mu)
    sizes  = St_to_r(Stokes, rho_gas,T_gas,rho_dust,  k_B,m_p,mu)
    masses = get_mass(sizes, rho_dust)


    return sizes,masses, T_gas
