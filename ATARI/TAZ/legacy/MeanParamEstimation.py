import math
import numpy as np
from scipy.special import gamma, gammaincc
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import FileReader

def wigner(X, MLS=1.0):
    c = math.pi/(4*MLS**2)
    return 2*c*X*np.exp(-c*X**2)
def plot_ecdf(X, E0=0.0, color='k'):
    N = len(X)
    plt.plot([E0, X[0]], [0.0, 0.0], f'-{color}')
    [plt.plot([X[idx-1],X[idx]], [idx/N, idx/N], f'-{color}') for idx in range(1,N)]
    [plt.plot([x, x], [idx/N, (idx+1)/N], f'-{color}') for idx, x in enumerate(X)]
def plot_wigner(LS, MLS=1.0, norm=True):
    if norm:
        LS /= MLS
        plt.hist(LS, density=True)
        X = np.linspace(0.0, np.max(LS), 1000)
        Y = wigner(X)
        # plt.plot(X, Y, '-k', linewidth=2)
    else:
        plt.hist(LS, density=True)
        X = np.linspace(0.0, np.max(LS), 1000)
        Y = wigner(X, MLS)
        # plt.plot(X, Y, '-k', linewidth=2)

res, SG_type = FileReader.readSammyPar('/Users/colefritsch/ENCORE/Python_ENCORE/SAMQUA.PAR')
E  = res.E
Gn = res.Gn
Gg = res.Gg
EA = E[SG_type==0]
EB = E[SG_type==1]

plt.figure(1)
plot_ecdf(E)
plt.tight_layout()
plt.show()

plt.figure(2)
plot_ecdf(EA, color='b')
# plot_wigner(np.diff(EA), )
# plt.tight_layout()
# plt.show()

# plt.figure(3)
plot_ecdf(EB, color='r')
# plot_wigner(np.diff(EB), )
plt.tight_layout()
plt.show()