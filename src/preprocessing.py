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

def r_to_m(r, rho_dust):
    """ Get particle mass from size and density.
        Assumes spherical shape for all particles.
    """
    return 4/3 * np.pi * rho_dust * r**3



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
    masses = r_to_m(sizes, rho_dust)

    return sizes,masses, T_gas



def MRN(a, imin,imax,rho_dust):
    """
    """
    da = a[-1]**0.5 - a[0]**0.5
    MRN = 0.5/r_to_m(da, rho_dust) * a**-3.5
    MRN[:imin] = 0
    MRN[imax:] = 0
    return MRN

def generate_inputs_basic(nbins,idxmin0,idxmax0 ,rho_gas,c_s):
    """ # First try: zero vrel, brownian only : check analytical solution dullemond/dominik 2005.
    """
    rho_dust = 1e1
    St_min, St_max = 0.01, 0.5
    Stokes = np.logspace(10**St_min,10**St_max,nbins)

    sizes, masses, T_gas = preprocessing_direct(Stokes,rho_dust,rho_gas,c_s)
    densities = MRN(sizes, sizes[idxmin0],sizes[idxmax0], rho_gas)
    velos = np.zeros((len(Stokes),3))

    return sizes, masses, densities, velos, T_gas



if __name__=="__main__":
    from matplotlib import pyplot as plt

    nbins    = 100
    imin     = 0
    imax     = 20
    rho_dust = 1e1

    sizes     = np.logspace(-3,-1,nbins)
    # sizes     = np.linspace(1e-5,5e-5,nbins)
    masses    = r_to_m(sizes,rho_dust)
    densities = MRN(sizes, imin,imax,rho_dust)
    
    y = densities * masses * sizes

    fig, ax = plt.subplots(1,1, figsize=(7,5))
    ax.plot(sizes,y, color="k")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"a [cm]")
    ax.set_ylabel(r"m $\cdot$ a $\cdot$ f(a) [g cm$^{-3}$]")
    plt.show()