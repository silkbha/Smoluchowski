import numpy as np

def solution_constant_kernel(m, a, S0, t):
    """ Analytical solution of the constant collision kernel R(m, m') = a.
        Initial condition is that only the zeroth mass bin is filled.
        
        Source: DustPy, Stammler & Birnstiel 2022.

        Parameters
        ----------
        m : Field
            Mass grid
        a : float
            Kernel constant
        S0 : float
            Total dust surface density
        t : float
            Time
        
        Returns
        -------
        Nm2 : Field
            Analytical solution in desired units.
    """
    m0 = m[0]
    N0 = S0 / m0
    return N0 / m0 * 4./(a*N0*t)**2 * np.exp( (1.-m/m0) * 2/(a*N0*t) ) * m**2

def solution_linear_kernel(m, a, S0, t):
    """ Analytical solution of the linear collision kernel R(m, m') = a(m + m').
        Initial condition is that only the zeroth mass bin is filled.

        Source: DustPy, Stammler & Birnstiel 2022.
        
        Parameters
        ----------
        m : Field
            Mass grid
        a : float
            Kernel constant
        S0 : float
            Total dust surface density
        t : float
            Time

        Returns
        -------
        Nm2 : Field
            Analytical solution in desired units.
    """
    m0 = m[0]
    N0 = S0/m0**2
    g = np.exp(-a*S0*t)
    #N = 1./m0**2 * g * np.exp( -m/m0 * ( 1. - np.sqrt(1-g) )**2 ) / ( 2.*np.sqrt(np.pi) * (m/m0)**1.5 * (1.-g)**0.75 )
    N = N0 * g * np.exp( -m/m0 * ( 1. - np.sqrt(1-g) )**2 ) / ( 2.*np.sqrt(np.pi) * (m/m0)**1.5 * (1.-g)**0.75 )
    return N*m**2