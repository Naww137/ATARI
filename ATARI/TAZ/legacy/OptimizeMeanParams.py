import math
import numpy as np

from RMatrix import ReduceFactor
from Distributions import Porter_Thomas_cdf

__doc__ = """
...
"""

def Anderson_Darling(X):
    n  = len(X)
    Xs = np.sort(X)
    return -n - np.mean(np.arange(1,2*n,2).reshape(-1,1) * np.log(Xs*(1.0-np.flip(Xs))))

poly = np.polynomial.polynomial.Polynomial([-4.72725274, -2.22442511, -6.01114371,  0.83633392, -6.35931921])
def rep_score(E, SGs, Freqs, Gnm):
    """
    ...
    """

    n = len(Freqs)

    # Getting Anderson-Darling statistic:
    A2 = np.zeros((n,))
    for g in range(n):                  # Spin group index
        ls = Freqs[0,g]*np.concatenate([np.diff(E[SGs[:,j] == g]) for j in range(SGs.shape[1])], axis=0)

        # Anderson-Darling statistic:
        R = np.exp(-math.pi/4*(ls*ls))
        A2[g] = Anderson_Darling(R)

    # Empirically-derived Anderson-Darling PDF:
    logA2 = np.log(A2)
    log_probs = poly(logA2)

    # Combining Probabilities over spin groups and epochs:
    return np.sum(log_probs)
    # # [print(f'P1 = {p1:.3e}') for p1 in P1]
    # combined_stat = np.mean(np.sum(log_probs, axis=1), axis=0)
    # return combined_stat

def rep_score_part(Res, SGs, MP):
    """
    ...
    """

    n = MP.Freq.shape[1]

    # Getting Anderson-Darling statistic:
    A2 = np.zeros((n,2))
    for g in range(n):                  # Spin group index

        # Level-Spacing:
        LS = []
        RGN = []
        for j in range(SGs.shape[1]):
            Idx = np.where(SGs[:,j] == g)
            LS.append(np.diff(Res.E[Idx]))

            E = Res.E[Idx]
            Gn = Res.Gn[Idx]
            rgn = Gn * ReduceFactor(E, MP.L[0,g], MP.A, MP.ac)
            RGN.append(rgn)
        ls = MP.Freq[0,g] * np.concatenate(LS, axis=0)
        rGn = np.concatenate(RGN, axis=0) / MP.Gnm[0,g]
        # ls = Freqs[0,g]*np.concatenate([np.diff(E[SGs[:,j] == g]) for j in range(SGs.shape[1])], axis=0)


        # Anderson-Darling statistic:
        R = np.exp(-math.pi/4*(ls*ls))
        A2[g,0] = Anderson_Darling(R)
        
        R = Porter_Thomas_cdf(rGn)
        A2[g,1] = Anderson_Darling(R)
        # print(A2[g])

        # print(f'group={g} | len={len(ls)}')

    # Empirically-derived Anderson-Darling PDF:
    logA2 = np.log(A2)
    log_probs = poly(logA2)

    # Combining Probabilities over spin groups and epochs:
    return log_probs






class GradDescent:
    """
    ...
    """

    def __init__(self, n:int=1, dtype='f4', alpha:float=1.0):
        """
        ...

        Nelder-Mead Method
        """

        self.n = n
        self.X = np.zeros((n,n+1), dtype=dtype)
        self.Y = np.zeros((1,n+1), dtype=dtype)

        self.alpha = alpha

    def value(self, x, y):
        if y >= self.Y[0,-1]:
            Xbad = x
            Ybad = y
        else:
            Xbad = self.X[:,-1]
            Ybad = self.Y[0,-1]
            self.X = np.c_((self.X[:,:-1], x))
            self.Y = np.c_((self.Y[:,:-1], y))
            idx = np.argsort(self.Y, axis=1)
            self.X = self.X[:,idx]
            self.Y = self.Y[:,idx]

        # Reflect:
        Xc = np.mean(self.X, axis=1)
        Xr = Xc + self.alpha * (Xc - Xbad)


        # ...

        self.X[:,:-1] = self.X[:,1:]
        self.Y[:,:-1] = self.Y[:,1:]
        self.X[:,-1]  = 












