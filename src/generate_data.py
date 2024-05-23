import os
from pathlib import Path
import itertools
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

def parameter_sweep(output_dir, kernel):
    """ TODO find way to save/store outputs within iterator
        TODO structure outputs, write save function
    """
    
    # Mass grid: stays constant throughout simulation
    m  = np.logspace(-12,3,100)

    # Floating point parameters to sweep
    a  = np.logspace(-6,2, 1000)
    S0 = np.logspace(-10,3,1000)
    t  = np.logspace(-9,3, 1e5)
    
    kernel_dict = {"constant": solution_constant_kernel, "linear": solution_linear_kernel}

    for i in itertools.product(a, S0, t):
        output = kernel_dict[kernel](m,*i)
    
    return

if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = str(Path(src_dir).parents[0])
    output_dir = os.path.join(main_dir, "data")
    print(f"Saving data in directory: {output_dir}")

    # kernel = "linear"
    kernel = "constant"
    parameter_sweep(output_dir, kernel)