import numpy as np

def E_to_T(E_gas):
    """ Converts kinetic energy of gas (as given by FARGO) to temperature using kinetic theory of gases.
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

def get_mass(r, rho):
    """ Assumes spherical shape for all dust grains.
    """
    return rho * 4/3 * np.pi * r**3



def preprocessing_direct(Stokes,gasstate):
    """ Takes state and info given by FARGO3D and converts to input for dust evolution step.


        # TODO convert 3D velocity distribution to 2D relative velocity matrix?
    """

    # Gas state
    rho_gas  = gasstate[0]      # single value
    E_gas    = gasstate[1]      # single value
    c_s      = gasstate[2]      # single value
    rho_dust = gasstate[3]      # single value
    specvol  = 1/rho_dust

    T_gas  = E_to_T(E_gas)
    sizes  = St_to_r(Stokes, rho_gas,T_gas,specvol)
    masses = get_mass(sizes, rho_dust)


    return sizes,masses, T_gas
