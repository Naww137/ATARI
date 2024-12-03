from collections.abc import Callable
import numpy as np
from pandas import DataFrame
from copy import copy

from TAZ import Reaction

__doc__ = """
This file contains all information regarding the MissingBayes methods, which calculates probability
density functions for resonances at some energy with some spingroup, given the known ladder.
Additionally, this distribution can be sampled probabilistically.

WORK IN PROGRESS!
"""

def missing_pdf(E, EL:float, ER:float, pdf:Callable, pM:float):
    """
    ...
    """

    return (pM / ((1-pM)*pdf(ER-EL))) * pdf(E-EL) * pdf(ER-E)

class MissBayes:
    """
    ...
    """

    def __init__(self, reaction:Reaction, resonances:DataFrame, err:float=0.005):
        # self.reaction = reaction
        self.resonances = resonances
        self.distributions = reaction.distributions('Missing', err=err)
        self.no_missing_dists = reaction.distributions('Wigner', err=err)
        self.pM = reaction.MissFrac
        self.energy_boundaries = reaction.EB
        self.num_groups = reaction.num_groups

    def missing_pdf(self, E, g:int):
        """
        ...
        """

        E_res = self.resonances.loc[self.resonances['J_ID'] == g+1, 'E'].to_numpy()
        distribution = self.distributions[g]
        pM = self.pM[g]
        coef = pM/(1-pM)

        probabilities = np.zeros((len(E),))
        indices = np.searchsorted(E_res, E)
        
        # Center cases:
        centers = (indices != 0) & (indices != len(E_res))
        center_indices = indices[centers]
        dx1 = E[centers] - E_res[center_indices-1]
        dx2 = E_res[center_indices] - E[centers]
        dxT = E_res[center_indices] - E_res[center_indices-1]
        probabilities[centers] = coef * distribution.f0(dx1) * distribution.f0(dx2) / distribution.f0(dxT)

        # Left edge case:
        lefts = (indices == 0)
        dx1 = E[lefts] - self.energy_boundaries[0]
        dx2 = E_res[0] - E[lefts]
        dxT = E_res[0] - self.energy_boundaries[0]
        probabilities[lefts] = coef * distribution.f1(dx1) * distribution.f0(dx2) / distribution.f1(dxT)

        # Right edge case:
        rights = (indices == len(E_res))
        dx1 = E[rights] - E_res[-1]
        dx2 = self.energy_boundaries[1] - E[rights]
        dxT = self.energy_boundaries[1] - E_res[-1]
        probabilities[rights] = coef * distribution.f0(dx1) * distribution.f1(dx2) / distribution.f1(dxT)

        return probabilities

    # def missing_pdf(self, E, g:int):
    #     """
    #     ...
    #     """

    #     E_res = self.resonances.loc[self.resonances['J_ID'] == g+1, 'E'].to_numpy()
    #     distribution = self.distributions[g]
    #     pM = self.pM[g]

    #     indices = np.searchsorted(E_res, E)
    #     if np.any(indices == 0) or np.any(indices == len(E_res)):
    #         raise NotImplementedError('Have not implemented edge cases yet.')
    #     dx1 = E - E_res[indices-1]
    #     dx2 = E_res[indices] - E
    #     dxT = E_res[indices] - E_res[indices-1]

    #     return (pM/(1-pM)) * distribution.f0(dx1) * distribution.f0(dx2) / distribution.f0(dxT)
    
    def _prob_no_miss(self, x:float, g:int):
        """
        ...
        """

        pM = self.pM[g]
        miss_dist = self.distributions[g]
        no_miss_dist = self.no_missing_dists[g]
        return (1-pM) * no_miss_dist.pdf(x) / miss_dist.pdf(x)
    
    def MissSample(self, trials:int=1, rng=None, seed:int=None):
        """
        ...
        """

        # L1) Loop over spingroup.
        # L2) Loop over area between resonances of the spingroup (while loop since the number of resonances is changing).
        # L3) Loop until break.
        # 1) Integrate PDF on the domain between resonances. Define as `I`.
        # 2) Break loop with probability `exp(-I)`.
        # 3) Sample one energy weighted by the PDF.
        # 4) Add resonance energy to the spingroup's resonance ladder.
        # 5) Continue L3.
        # 6) Continue L2.
        # 7) Continue L1.
        # 8) Return new resonance ladder.

        if rng is None:     rng = np.random.default_rng(seed)

        # resonances_new = copy(self.resonances)
        for g in range(self.num_groups):
            E_res_g = self.resonances.loc[self.resonances['J_ID'] == g+1, 'E'].to_numpy()
            E_res_new_g = copy(E_res_g)
            pM = self.pM[g]
            num_energies_lim = int(np.ceil(len(E_res_g) / (1 - pM)**5))
            for i in range(num_energies_lim-1):
                if i >= len(E_res_new_g) - 1:
                    break # finished loop
                while True:
                    Delta = E_res_new_g[i+1] - E_res_new_g[i]
                    prob_no_miss = self._prob_no_miss(Delta, g)
                    if rng.random() <= prob_no_miss:
                        break # break if no missing resonances should be added; move up to next region.
                    # ...
                    raise NotImplementedError('...')
            else:
                raise RuntimeError('...')