"""
    Numerical "brute-force" implementation of the Smoluchowski equation for coagulation and fragmentation of colliding solid particles.
    
"""

import numpy as np

def sum_to_n(n, size=2, limit=None):
    """Produce all lists of `size` positive integers in decreasing order
    that add up to `n`."""
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

def kernel_coag(i,j):
    return 0.5

def kernel_frag(i,j):
    return 0.5

def Smoluchowski(dust, dt=1):
    """ Single-step time evolution of the Smoluchowski equation.
    
        Input : array of mass distribution at time t0
            - flexible number of discrete mass bins
            - row 1/3 : mass corresponding to each bin
            - row 2/3 : mass density per mass bin
            - row 3/3 : particle velocity per mass bin
        
        Output : array of mass distribution at time t0 + dt
            - same mass bins as input array
            - row 1/1 : mass density per mass bin
        
        kwargs :
            - dt : size of timestep (later)

        #TODO:
        - Mass conservation in log bins? (start with linearly spaced bins, work from there)
        - Boundary conditions: coagulation with largest bin, fragmentation with smallest bin?
        - Coagulation/fragmentation kernels
        - Optimization :
            - Summation scheme: no double counting!
            - Vectorize where possible
            - numpy (C) vs native Python
    """

    masses = dust[0]
    densities = dust[1]
    velos = dust[2]

    densities_new = np.zeros([len(masses)])
    for i,(m_i,n_i,v_i) in enumerate(zip(masses,densities,velos)):
        dndt_i = 0
        dndt_i1 = 0
        for j,(m_j,n_j,v_j) in enumerate(zip(masses,densities,velos)):
            vrel = np.abs(v_i - v_j)
            dndt_i += kernel_coag(i,j)*n_i*n_j
            if i+j < len(masses):
                n_ij = densities[i+j] #n_{i+j}
                dndt_i -= kernel_frag(i,j)*n_ij
            if m_j < m_i:
                n_ji = densities[i-j] #n_{i-j}
                dndt_i1 += kernel_coag(i-j,j)*n_i*n_ji-kernel_frag(i-j,j)*n_i
        densities_new[i] = n_i + dndt_i + 0.5*dndt_i1
    
    return densities_new



if __name__=="__main__":
    print("\n###################### Input #######################")
    masses = np.logspace(-5,3,30)
    densities = np.linspace(0.2,0.5,30)
    velos = np.linspace(0.5,0.1,30)
    test = np.array([masses,densities,velos])
    print(test[1])
    
    print("\n##################### Output #######################")
    newtest = Smoluchowski(test)
    print(newtest)
    print("####################################################")