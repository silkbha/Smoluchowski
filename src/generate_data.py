import os
import sys
from pathlib import Path
import shutil

import numpy as np

# CONSTANT KERNEL
def solution_constant_kernel(t, m, a, S0):
    """Analytical solution of the constant collision kernel R(m, m') = a.
    Initial condition is that only the zeroth mass bin is filled.

    Parameters
    ----------
    t : float
        Time
    m : Field
        Mass grid
    a : float
        Kernel constant
    S0 : float
        Total dust surface density

    Returns
    -------
    Nm2 : Field
        Analytical solution in desired units."""
    m0 = m[0]
    N0 = S0 / m0
    return N0 / m0 * 4./(a*N0*t)**2 * np.exp( (1.-m/m0) * 2/(a*N0*t) ) * m**2


# LINEAR KERNEL
def solution_linear_kernel(t, m, a, S0):
    """Analytical solution of the linear collision kernel R(m, m') = a(m + m').
    Initial condition is that only the zeroth mass bin is filled.

    Parameters
    ----------
    t : float
        Time
    m : Field
        Mass grid
    a : float
        Kernel constant
    S0 : float
        Total dust surface density

    Returns
    -------
    Nm2 : Field
        Analytical solution in desired units."""
    m0 = m[0]
    N0 = S0/m0**2
    g = np.exp(-a*S0*t)
    #N = 1./m0**2 * g * np.exp( -m/m0 * ( 1. - np.sqrt(1-g) )**2 ) / ( 2.*np.sqrt(np.pi) * (m/m0)**1.5 * (1.-g)**0.75 )
    N = N0 * g * np.exp( -m/m0 * ( 1. - np.sqrt(1-g) )**2 ) / ( 2.*np.sqrt(np.pi) * (m/m0)**1.5 * (1.-g)**0.75 )
    return N*m**2

def parameter_sweep(output_dir, kernel):
    # T      = np.logspace(-9,3,100)
    # M      = np.logspace()
    # A      = np.linspace()
    # Sigmas = np.logspace()
    
    # solution_constant_kernel()
    # solution_linear_kernel()
    return

if __name__=="__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = str(Path(src_dir).parents[0])
    output_dir = os.path.join(main_dir, "data")
    print(f"Saving data in directory: {output_dir}")

    # kernel = "linear"
    kernel = "constant"
    parameter_sweep(output_dir, kernel)