




def preprocessing(dustinfo,duststate,gasstate):
    """ Takes state and info given by FARGO3D and converts to input for dust evolution step.


        # TODO convert 3D velocity distribution to 2D relative velocity matrix
    """


    # Dust Info
    Stokes = dustinfo           # (N,1) array

    # Dust state
    densities = duststate[:,0]  # (N,1) array
    velos = duststate[:,1:]     # (N,3) array

    # Gas state
    rho_gas  = gasstate[0]      # single value
    E_gas    = gasstate[1]      # single value
    rho_dust = gasstate[2]      # single value

    return sizes,masses,vrel
