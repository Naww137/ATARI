
from ATARI.theory.resonance_statistics import sample_RRR_levels, sample_RRR_widths
import numpy as np
import pandas as pd





def sample_resonance_ladder(Erange, spin_groups, 
                            ensemble='NNE',
                            rng=None, seed=None):
        """
        Samples a full resonance ladder.

        _extended_summary_

        Parameters
        ----------
        Erange : array-like
            _description_
        spin_groups : list
            List of tuples defining the spin groups being considered.
        average_parameters : Dictionary
            Dictionary containing the average resonance parameters and degrees of freedom for each reaction and spin group .
        ensemble : "NNE", "GOE", "GUE", "GSE", or "Poisson"
            The level-spacing distribution to sample from:
            NNE : Nearest Neighbor Ensemble
            GOE : Gaussian Orthogonal Ensemble
            GUE : Gaussian Unitary Ensemble
            GSE : Gaussian Symplectic Ensemble
            Poisson : Poisson Ensemble
        rng : np.random.Generator or None
            Numpy random number generator object. Default is None.
        seed : int or None
            Random number generator seed. Only used when rng is None. Default is None.

        Returns
        -------
        DataFrame
            Resonance ladder information.
        """
        # Random number generator:
        if rng is None:
            if seed is None:
                rng = np.random # uses np.random.seed
            else:
                rng = np.random.default_rng(seed) # generates rng from provided seed

        # resonance_ladder = pd.DataFrame()
        resonance_ladder = pd.DataFrame({'E':[], 'Gg':[], 'Gn1':[], 'J':[], 'J_ID':[]})

        for Jpi, Jinfo in spin_groups.items():

            # sample resonance levels for each spin group with negative parity
            [levels, level_spacing] = sample_RRR_levels(Erange, Jinfo["<D>"], ensemble=ensemble, rng=rng)
            
            # if no resonance levels sampled
            if len(levels) == 0:
                continue
            # elif len(levels) == 1:

            # a single radiative capture width is sampled w/large DOF because of many 'partial' radiative transitions to ground state
            # must divide average by the 2 in order to maintain proper distirbution b/c we sample on red_gwidth
            red_gwidth = sample_RRR_widths(levels, Jinfo["<Gg>"]/2, Jinfo["g_dof"], rng=rng)
            Gwidth = 2*red_gwidth # Gbar = 2*gbar b/c P~1 for gamma channels

            # sample observable width as sum of multiple single-channel width with the same average (chi2, DOF=channels)
            nwidth = sample_RRR_widths(levels, Jinfo["<Gn>"], Jinfo["n_dof"], rng=rng)
            E_Gn_gnx2 = pd.DataFrame([levels, Gwidth, nwidth, np.zeros(len(levels)), np.zeros(len(levels)), np.zeros(len(levels)),
                                      [Jpi]*len(levels), [Jinfo['J_ID']]*len(levels)], index=['E', 'Gg', 'Gn1', 'varyE', 'varyGg', 'varyGn1', 'J', 'J_ID'])
            # assert len(np.unique(j[2]))==1, "Code cannot consider different l-waves contributing to a spin group"
            resonance_ladder = pd.concat([resonance_ladder, E_Gn_gnx2.T])

        resonance_ladder.reset_index(inplace=True, drop=True)
    
        return resonance_ladder



def sample_resonance_ladder_old(Erange, spin_groups, average_parameters, 
                                ensemble='NNE',
                                rng=None, seed=None):
        """
        Samples a full resonance ladder.

        _extended_summary_

        Parameters
        ----------
        Erange : array-like
            _description_
        spin_groups : list
            List of tuples defining the spin groups being considered.
        average_parameters : DataFrame
            DataFrame containing the average resonance parameters for each spin group.
        ensemble : "NNE", "GOE", "GUE", "GSE", or "Poisson"
            The level-spacing distribution to sample from:
            NNE : Nearest Neighbor Ensemble
            GOE : Gaussian Orthogonal Ensemble
            GUE : Gaussian Unitary Ensemble
            GSE : Gaussian Symplectic Ensemble
            Poisson : Poisson Ensemble
        rng : np.random.Generator or None
            Numpy random number generator object. Default is None.
        seed : int or None
            Random number generator seed. Only used when rng is None. Default is None.

        Returns
        -------
        DataFrame
            Resonance ladder information.
        """
        # Random number generator:
        if rng is None:
            if seed is None:
                rng = np.random # uses np.random.seed
            else:
                rng = np.random.default_rng(seed) # generates rng from provided seed

        # resonance_ladder = pd.DataFrame()
        resonance_ladder = pd.DataFrame({'E':[], 'Gg':[], 'gn2':[], 'J':[], 'chs':[], 'lwave':[], 'J_ID':[]})
        J_ID = 0
        for ij, j in enumerate(spin_groups):
            J_ID += 1

            # sample resonance levels for each spin group with negative parity
            [levels, level_spacing] = sample_RRR_levels(Erange, average_parameters.dE[f'{j[0]}'], ensemble=ensemble, rng=rng)
            
            # if no resonance levels sampled
            if len(levels) == 0:
                continue
            # elif len(levels) == 1:


            # a single radiative capture width is sampled w/large DOF because of many 'partial' radiative transitions to ground state
            # must divide average by the 2*DOF in order to maintain proper magnitude
            red_gwidth = sample_RRR_widths(levels, average_parameters.Gg[f'{j[0]}']/2, 1000, rng=rng)
            Gwidth = 2*red_gwidth # Gbar = 2*gbar b/c P~1 for gamma channels

            # sample observable width as sum of multiple single-channel width with the same average (chi2, DOF=channels)
            red_nwidth = sample_RRR_widths(levels, average_parameters.gn2[f'{j[0]}']/j[1], j[1], rng=rng)
            E_Gn_gnx2 = pd.DataFrame([levels, Gwidth, red_nwidth, [j[0]]*len(levels), [j[1]]*len(levels), [j[2]]*len(levels), [J_ID]*len(levels)], index=['E','Gg', 'gn2', 'J', 'chs', 'lwave', 'J_ID'])  
            # assert len(np.unique(j[2]))==1, "Code cannot consider different l-waves contributing to a spin group"
            resonance_ladder = pd.concat([resonance_ladder, E_Gn_gnx2.T])

        resonance_ladder.reset_index(inplace=True, drop=True)
    
        return resonance_ladder