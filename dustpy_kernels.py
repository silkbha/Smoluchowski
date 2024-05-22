import numpy as np


def convert(S, m):
    """Function converts the Dustpy units into number densities.

    Parameters
    ----------
    S : array
        Integrated surface density in DustPy units
    m : array
        mass grid

    Returns
    -------
    Nm2 : array
        Simulation results in desired units for comparison"""
    A = np.mean(m[1:]/m[:-1])
    B = 2 * (A-1) / (A+1)
    return S / B

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



def set_constant_kernel(sim, a, S0):
    """Function set the ``DustPy`` simulation object up for the constant collision kernel R(m, m') = a.

    sim : Frame
        Simulation object
    a : float
        Kernel constant
    S0 : float
        Total dust surface density"""
    # Turning off gas evolution by removing integrator instruction
    del(sim.integrator.instructions[1])
    # Turning off gas source to not influence the time stepping
    sim.gas.S.tot[...] = 0.
    sim.gas.S.tot.updater = None
    # Turning off dust advection
    sim.dust.v.rad[...] = 0.
    sim.dust.v.rad.updater = None
    # Turning off fragmentation. Only sticking is considered
    sim.dust.p.frag[...] = 0.
    sim.dust.p.frag.updater = None
    sim.dust.p.stick[...] = 1.
    sim.dust.p.stick.updater = None
    # Setting the constant kernel
    sim.dust.kernel[...] = a
    sim.dust.kernel.updater = None
    # Setting the initial time
    sim.t = 1.e-9
    # Setting the initial dust surface density
    m = sim.grid.m
    A = np.mean(m[1:]/m[:-1])
    B = 2 * (A-1) / (A+1)
    sim.dust.Sigma[...] = sim.dust.SigmaFloor[...]
    sim.dust.Sigma[1, :] = np.maximum(solution_constant_kernel(sim.t, m, a, S0)*B, sim.dust.SigmaFloor[1, :])
    # Updating the simulation object
    sim.update()

def set_linear_kernel(sim, a, S0):
    """Function set the ``DustPy`` simulation object up for the linear collision kernel R(m, m') = a(m + m').

    sim : Frame
        Simulation object
    a : float
        Kernel constant
    S0 : float
        Total dust surface density"""
    # Turning off gas evolution by removing integrator instruction
    del(sim.integrator.instructions[1])
    # Turning off gas source to not influence the time stepping
    sim.gas.S.tot[...] = 0.
    sim.gas.S.tot.updater = None
    # Turning off dust advection
    sim.dust.v.rad[...] = 0.
    sim.dust.v.rad.updater = None
    # Turning off fragmentation. Only sticking is considered
    sim.dust.p.frag[...] = 0.
    sim.dust.p.frag.updater = None
    sim.dust.p.stick[...] = 1.
    sim.dust.p.stick.updater = None
    # Setting the constant kernel
    sim.dust.kernel[...] = a * (sim.grid.m[:, None] + sim.grid.m[None, :])[None, ...]
    sim.dust.kernel.updater = None
    # Setting the initial time
    sim.t = 1.0
    # Setting the initial dust surface density
    m = sim.grid.m
    A = np.mean(m[1:]/m[:-1])
    B = 2 * (A-1) / (A+1)
    sim.dust.Sigma[...] = sim.dust.SigmaFloor[...]
    sim.dust.Sigma[1, ...] = np.maximum(solution_linear_kernel(sim.t, m, a, S0) * B, sim.dust.SigmaFloor[1, ...])
    # Updating the simulation object
    sim.update()

