
from ATARI.syndat.sample_levels import sample_RRR_levels
from ATARI.syndat.sample_widths import sample_RRR_widths
import numpy as np
import pandas as pd





def sample_resonance_ladder(Erange, spin_groups, average_parameters, 
                                                            use_fudge=False):
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
        use_fudge : bool, optional
            Option to use Syndat for resonance sampling or the higher-fidelity implementation in Fudge.
            The latter option is dependent on a user install of the Fudge code, by default False.

        Returns
        -------
        DataFrame
            Resonance ladder information.
        """

        # TODO implement option to use Fudge for resonance sampling. Calling a single function would be nice. I will clean up the 'else' option to also be a single function
        if use_fudge:
            raise ValueError("Need to implement this option")
        else:
            # resonance_ladder = pd.DataFrame()
            resonance_ladder = pd.DataFrame({'E':[], 'Gg':[], 'Gn':[], 'J':[], 'chs':[], 'lwave':[], 'J_ID':[]})
            J_ID = 0
            for ij, j in enumerate(spin_groups):
                
                J_ID += 1

                # sample resonance levels for each spin group with negative parity
                [levels, level_spacing] = sample_RRR_levels(Erange, average_parameters[str(j[0])]["<D>"])
                
                # if no resonance levels sampled
                if len(levels) == 0:
                    continue
                # elif len(levels) == 1:


                # a single radiative capture width is sampled w/large DOF because of many 'partial' radiative transitions to ground state
                # must divide average by the 2 in order to maintain proper distirbution b/c we sample on red_gwidth
                red_gwidth = sample_RRR_widths(levels, average_parameters[str(j[0])]["<Gg>"]/2, average_parameters[str(j[0])]["g_dof"])
                Gwidth = 2*red_gwidth # Gbar = 2*gbar b/c P~1 for gamma channels

                # sample observable width as sum of multiple single-channel width with the same average (chi2, DOF=channels)
                nwidth = sample_RRR_widths(levels, average_parameters[str(j[0])]["<Gn>"], average_parameters[str(j[0])]["g_dof"])
                E_Gn_gnx2 = pd.DataFrame([levels, Gwidth, nwidth, [j[0]]*len(levels), [j[1]]*len(levels), [j[2]]*len(levels), [J_ID]*len(levels)], index=['E','Gg', 'Gn', 'J', 'chs', 'lwave', 'J_ID'])  
                # assert len(np.unique(j[2]))==1, "Code cannot consider different l-waves contributing to a spin group"
                resonance_ladder = pd.concat([resonance_ladder, E_Gn_gnx2.T])

            resonance_ladder.reset_index(inplace=True, drop=True)
    
        return resonance_ladder



def sample_resonance_ladder_old(Erange, spin_groups, average_parameters, 
                                                                        use_fudge=False):
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
        use_fudge : bool, optional
            Option to use Syndat for resonance sampling or the higher-fidelity implementation in Fudge.
            The latter option is dependent on a user install of the Fudge code, by default False.

        Returns
        -------
        DataFrame
            Resonance ladder information.
        """

        # TODO implement option to use Fudge for resonance sampling. Calling a single function would be nice. I will clean up the 'else' option to also be a single function
        if use_fudge:
            raise ValueError("Need to implement this option")
        else:
            # resonance_ladder = pd.DataFrame()
            resonance_ladder = pd.DataFrame({'E':[], 'Gg':[], 'gn2':[], 'J':[], 'chs':[], 'lwave':[], 'J_ID':[]})
            J_ID = 0
            for ij, j in enumerate(spin_groups):
                J_ID += 1

                # sample resonance levels for each spin group with negative parity
                [levels, level_spacing] = sample_RRR_levels(Erange, average_parameters.dE[f'{j[0]}'])
                
                # if no resonance levels sampled
                if len(levels) == 0:
                    continue
                # elif len(levels) == 1:


                # a single radiative capture width is sampled w/large DOF because of many 'partial' radiative transitions to ground state
                # must divide average by the 2*DOF in order to maintain proper magnitude
                red_gwidth = sample_RRR_widths(levels, average_parameters.Gg[f'{j[0]}']/2, 1000)
                Gwidth = 2*red_gwidth # Gbar = 2*gbar b/c P~1 for gamma channels

                # sample observable width as sum of multiple single-channel width with the same average (chi2, DOF=channels)
                red_nwidth = sample_RRR_widths(levels, average_parameters.gn2[f'{j[0]}']/j[1], j[1])
                E_Gn_gnx2 = pd.DataFrame([levels, Gwidth, red_nwidth, [j[0]]*len(levels), [j[1]]*len(levels), [j[2]]*len(levels), [J_ID]*len(levels)], index=['E','Gg', 'gn2', 'J', 'chs', 'lwave', 'J_ID'])  
                # assert len(np.unique(j[2]))==1, "Code cannot consider different l-waves contributing to a spin group"
                resonance_ladder = pd.concat([resonance_ladder, E_Gn_gnx2.T])

            resonance_ladder.reset_index(inplace=True, drop=True)
    
        return resonance_ladder