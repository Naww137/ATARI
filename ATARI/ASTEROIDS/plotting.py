import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ATARI.ModelData.particle_pair import Particle_Pair

__doc__ = """
This file contains plotting tools to assist with verification of mean parameter estimation.
"""

def plot_ecdf(energies, bounds:tuple, title:str=None, figname:str=None):
    """
    ...
    """

    energies_sorted = np.sort(energies)[energies > bounds[0]]
    energies_cut = energies_sorted
    energies_cut = energies_cut[energies_cut < bounds[1]]
    N = len(energies_cut)
    x = np.concatenate(([bounds[0]], energies_cut, [bounds[1]]))
    dx = np.diff(x)
    dx2 = np.diff(x*x)
    y = np.arange(N+1)
    Delta1 = bounds[1]    - bounds[0]
    Delta2 = bounds[1]**2 - bounds[0]**2
    Delta3 = bounds[1]**3 - bounds[0]**3

    a = np.sum(y*dx)
    b = np.sum(y*dx2)
    A = 6*(b*Delta1 - a*Delta2) / (4*Delta1*Delta3 - 3*Delta2*Delta2)

    E_mid = 0.5*(bounds[0] + bounds[-1])
    B = a/Delta1 - E_mid*A
    # B = 0
    print(1/A)

    plt.figure()
    plt.clf()
    plt.axvline(bounds[ 0], color='red', linestyle=':')
    plt.axvline(bounds[-1], color='red', linestyle=':')
    plt.plot([energies_sorted[0], energies_sorted[-1]], [A*energies_sorted[0]+B, A*energies_sorted[-1]+B], '-b', label='Linear Fit')
    # plt.ecdf(energies, weights=np.ones_like(energies), color='black')
    y = np.arange(1, len(energies_sorted)+1)
    plt.step(energies_sorted, y, where='post', color='black')
    plt.xlabel('Energy (eV)', fontsize=16)
    plt.ylabel('Cumulative Level Density', fontsize=16)
    if title is not None:
        plt.title(title, fontsize=18)
    plt.tight_layout()
    if figname is None:     plt.show()
    else:                   plt.savefig(figname)


def plot_Porter_Thomas_survival(res_ladder:pd.DataFrame, particle_pair:Particle_Pair, title:str=None, figname:str=None):
    """
    ...
    """

    plt.figure()
    plt.clf()
    
    if title is not None:
        plt.title(title, fontsize=18)
    plt.tight_layout()
    if figname is None:     plt.show()
    else:                   plt.savefig(figname)