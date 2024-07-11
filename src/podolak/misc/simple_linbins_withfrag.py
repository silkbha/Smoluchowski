"""
    Numerical "brute-force" implementation of the Smoluchowski-like equation for coagulation and fragmentation of colliding solid particles.
    
"""

import numpy as np

def sum_to_n(n, size=2, limit=None):
    """Produce all lists of 'size' positive integers in decreasing order that add up to n.
    """
    if size == 1:
        yield [n]
        return
    if limit is None:
        limit = n
    start = (n + size - 1) // size
    stop = min(limit, n - size + 1) + 1
    for i in range(start, stop):
        for tail in sum_to_n(n - i, size - 1, i):
            yield [i] + tail

def v_brown(i,j):
    """ Brownian motion component to be added to relative particle velocities.
        Allows for treatment of collisions between same-sized particles
        (otherwise their relative velocities would be zero -> no collisions -> no coagulation/fragmentation).
    """
    return 0

def kernel_coag(i,j):
    """ Kernel for coagulation of dust grains.
    """
    return 1

def kernel_cfrag(i,j):
    """ Kernel for collisional fragmentation of dust grains.
    """
    return 1

def kernel_lfrag(i):
    """ Kernel for collisionless (i.e. linear) fragmentation of dust grains,
        due to e.g. gas friction, temperature, radiation.
        Set to 0 for now ==> ignore collisionless fragmentation
    """
    return 0

def Smoluchowski(dust, dt=1):
    """ Single-step time evolution of the Smoluchowski equation.
    
        Input : array of size distribution at time t0
            - flexible number of discrete size bins
            - row 1/3 : Stokes number corresponding to each bin
            - row 2/3 : mass density per size bin
            - row 3/3 : particle velocity per size bin
        
        Output : array of mass distribution at time t0 + dt
            - same size bins as input array
            - row 1/1 : mass density per size bin
        
        kwargs :
            - dt : size of timestep (later)

        #TODO:
        - Mass conservation in log bins? (start with linearly spaced bins, work from there)
        - Boundary conditions: coagulation with largest bin, fragmentation with smallest bin?
        - Coagulation/fragmentation kernels
        - Optimization :
            - Summation scheme: no double counting! (right now : double counting with factor 0.5 afterward)
            - Vectorize where possible
            - numpy (C) vs native Python
    """

    St = dust[0]
    densities = dust[1]
    velos = dust[2]

    densities_new = np.zeros([len(St)])
    for i,(s_i,n_i,v_i) in enumerate(zip(St,densities,velos)):
        
        # Mass loss due to linear fragmentation: friction with gas, temperature, radiation
        dndt_i = -n_i * kernel_lfrag(i)

        # Collisions between grains i & j
        dndt_i1 = 0
        for j,(s_j,n_j,v_j) in enumerate(zip(St,densities,velos)):
            
            # Define relative velocity
            #TODO: add to kernel (vrel does nothing right now)
            #TODO: incorporate 3D relative velocity
            # vrel = np.linalg.norm(v_i - v_j) + v_brown(s_i,s_j)
            vrel = np.abs(v_i - v_j) + v_brown(s_i,s_j)

            # Mass loss due to coagulation into m > m_i grains
            dndt_i -= n_i*n_j * kernel_coag(i,j)
            # Mass gain due to fragmentation of larger grains (no fragmentation of grains larger than max m_j)
            if i+j < len(St):
                n_ij = densities[i+j] #n_{i+j}
                dndt_i += n_ij * kernel_cfrag(i,j)
             
            # Interactions with smaller grains
            if s_j < s_i:
                n_ji = densities[i-j] #n_{i-j}
                # Mass gain due to coagulation of smaller grains into grains with m_i
                dndt_i1 += n_i*n_ji * kernel_coag(i-j,j)
                # Mass loss due to fragmentation of equal size grains
                dndt_i1 -= n_i * kernel_cfrag(i-j,j)

        # Add dndt to previous n_i(t) for new n_i(t+1) (factor 0.5 in i1-term to prevent double counting)
        densities_new[i] = n_i + dndt_i + 0.5*dndt_i1
    
    return densities_new


if __name__=="__main__":
    print("\n###################### Input #######################")
    St = np.logspace(-5,1,30)
    densities = np.linspace(0.1,1,30)
    velos = np.linspace(0.5,0.5,30)
    test = np.array([St,densities,velos])
    print(test[1])
    
    print("\n##################### Output #######################")
    newtest = Smoluchowski(test)
    print(newtest)
    print("####################################################")