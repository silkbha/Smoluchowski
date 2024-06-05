import time
import os
from pathlib import Path
import h5py
import itertools

import numpy as np
from coag_kernels import *




def parameter_sweep(output_dir, kernel):
    """ 
    """
    
    # Mass grid: stays constant throughout simulation
    m  = np.logspace(-12,4,100)

    # Floating point parameters to sweep
    a  = np.array([1.]) # np.linspace(0.1,10,50) #
    S0 = np.logspace(-4,3,200)
    t  = np.linspace(1e-9,1e3,20000)
    
    par_space = len(a)*len(S0)*len(t)
    print("Param space size:", len(a)*len(S0)*len(t))

    kernel_dict = {"constant": solution_constant_kernel, "linear": solution_linear_kernel}

    print("Creating iterable list of parameter value combinations. This may take a while...")
    inputs = itertools.product(a, S0, t)

    print("Sweeping parameter space...")
    cols = len(m) + 3
    data = np.zeros((par_space, cols))
    for i,input in enumerate(inputs):
        data[i,0:3] = np.array([*input])
        data[i,3:]  = kernel_dict[kernel](m,*input)
    
    filename = f"{output_dir}/data_{kernel}"
    with h5py.File(filename+".h5", "w") as hf:
        hf.create_dataset(f"data_{kernel}", data=data)
    # np.savetxt(filename+"dat", data)
    print("Done and Saved!")

    return

if __name__ == "__main__":
    start = time.time()

    src_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = str(Path(src_dir).parents[0])
    output_dir = os.path.join(main_dir, "data")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving data in directory: {output_dir}")

    parameter_sweep(output_dir, "constant")
    parameter_sweep(output_dir, "linear")

    end = time.time()
    print(f"Elapsed time: {end-start} s")