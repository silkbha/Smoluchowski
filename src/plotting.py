import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['lines.linewidth'] = 1.25

def plot_set(m, analytical, lstm_results, kernel, model_dir, onestep=False):
    if onestep:
        namestring = "onestep"
    else:
        namestring = "independent"

    fig, ax = plt.subplots(1,1, figsize=(7,5))
    
    idxs = np.linspace(0,99,7, dtype=np.int16)
    # idxs = [1,10,30,40,60,95]
    couleurs = plt.cm.inferno(np.linspace(0,1,len(idxs)+1))
    for i,idx in enumerate(idxs):
        ax.loglog(m, analytical[idx], c=couleurs[i], lw=1, ls="-.")
        ax.loglog(m, lstm_results[idx], c=couleurs[i])

    ax.set_xlim(m[0], 1e5)
    ax.set_ylim(1.e-6, 1.e3)
    ax.set_xlabel(r"$m$", math_fontfamily='dejavuserif')
    ax.set_ylabel(r"$N\,\left(m,t\right)\,\cdot\,m^2$", math_fontfamily='dejavuserif')
    if kernel == "constant":
        ax.set_title(r"Neural Network -- Constant Kernel: $M\left( m, m'\right) = 1$", math_fontfamily='dejavuserif')
    elif kernel == "linear":
        ax.set_title(r"Neural Network -- Linear Kernel: $M\left( m, m'\right) = m + m'$", math_fontfamily='dejavuserif')
    
    fig.tight_layout()

    imgname = f"plots/{namestring}.png"
    plt.savefig(os.path.join(model_dir,imgname), dpi=300)
    plt.close()
    # plt.show()


def plot_everything(m, analytical, lstm_results, kernel, model_dir, onestep=False):
    if onestep:
        namestring = "onestep"
    else:
        namestring = "indep"
    
    if kernel == "constant":
        kernelstring = "Constant"
        clr = "tab:blue"
    elif kernel == "linear":
        kernelstring = "Linear"
        clr = "tab:orange"

    for idx in range(100):
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        
        ax.loglog(m, analytical[idx], c="k", lw=1, ls="-.")
        ax.loglog(m, lstm_results[idx], c=clr)

        # ax.set_xlim(m[0, 0], m[0, -1])
        ax.set_ylim(1.e-30, 1.e3)
        ax.set_xlabel(r"$m$", math_fontfamily='dejavuserif')
        ax.set_ylabel(r"$N\,\left(m,t\right)\,\cdot\,m^2$", math_fontfamily='dejavuserif')
        ax.set_title(f"Neural Network -- {kernelstring} Kernel: t = {idx+1} / 100", math_fontfamily='dejavuserif')
        
        fig.tight_layout()

        imgname = f"plots/{namestring}_movie_t{str(idx).rjust(3, '0')}.png"
        plt.savefig(os.path.join(model_dir,imgname))
        plt.close()
        # plt.show()