import numpy as np
from ATARI.theory import scattering_params


def SLBW(E, pair, resonance_ladder):
    """
    _summary_

    _extended_summary_

    Parameters
    ----------
    E : _type_
        _description_
    pair : _type_
        _description_
    resonance_ladder : _type_
        _description_

    Returns
    -------
    xs_tot, xs_scat, xs_cap
        Vectors of total, scattering, and capture cross sections.
    """
    
    # TODO: check types in resonance ladder

    if resonance_ladder.empty:
        
        # window potential scattering
        lwave=0; Jpi=3.0
        window_shift, window_penetration, window_phi, window_k = scattering_params.FofE_explicit(E, pair.ac, pair.M, pair.m, lwave)
        g = scattering_params.gstat(Jpi, pair.I, pair.i)
        potential_scattering = (4*np.pi*g/(window_k**2)) * np.sin(window_phi)**2
        # TODO: if resonance ladder is empty, need to calculate scattering phase shift for EACH JPI
        xs_cap = 0
        xs_scat = potential_scattering
        xs_tot = xs_scat + xs_cap
        return xs_tot, xs_scat, xs_cap
        
    xs_cap = 0; xs_scat = 0
    group_by_J = dict(tuple(resonance_ladder.groupby('J')))

    for Jpi in group_by_J:
        
        J_df = group_by_J[Jpi]
        # assert J > 0 
        Jpi = abs(Jpi)

        # orbital_angular_momentum = J_df.lwave.unique()
        orbital_angular_momentum = np.unique(J_df.lwave.values[0]) # just takes the first level's l-wave vector, all should be the same in each group
        assert len(orbital_angular_momentum) == 1, "Cannot handle different l-waves contributing to multichannel widths"
        
        # calculate functions of energy -> shift, penetrability, phase shift
        g = scattering_params.gstat(Jpi, pair.I, pair.i) #(2*J+1)/( (2*ii+1)*(2*I+1) );   # spin statistical factor g sub(j alpha)
        shift, penetration, phi, k = scattering_params.FofE_explicit(E, pair.ac, pair.M, pair.m, orbital_angular_momentum[0])

        # calculate capture
        sum1 = 0
        for index, row in J_df.iterrows():
            E_lambda = row.E
            Gg = row.Gg * 1e-3
            # gnx2 = sum([row[ign] for ign in range(2,len(row))]) * 1e-3  # Not sampling multiple, single-channel particle widths
            gnx2 = row.gnx2 * 1e-3 
            Gnx = 2*penetration*gnx2

            d = (E-E_lambda)**2 + ((Gg+Gnx)/2)**2 
            sum1 += (Gg*Gnx) / ( d )

        xs_cap += (np.pi*g/(k**2))*sum1


        # calculate scatter
        sum1 = 0
        sum2 = 0
        sum3 = 0
        for index, row in J_df.iterrows():
            E_lambda = row.E
            Gg = row.Gg * 1e-3
            # gnx2 = sum([row[ign] for ign in range(2,len(row))]) * 1e-3 # Not sampling multiple, single-channel particle widths
            gnx2 = row.gnx2 * 1e-3 
            Gnx = 2*penetration*gnx2

            Gtot = Gnx+Gg
            d = (E-E_lambda)**2 + ((Gg+Gnx)/2)**2 
            sum1 += Gnx*Gtot/d
            sum2 += Gnx*(E-E_lambda)/d
            sum3 += (Gnx*(E-E_lambda)/d)**2 + (Gnx*Gtot/d/2)**2

        xs_scat += (np.pi*g/(k**2))* ( (1-np.cos(2*phi))*(2-sum1) + 2*np.sin(2*phi)*sum2 + sum3 )

    # calculate total
    xs_tot = xs_cap+xs_scat

    return xs_tot, xs_scat, xs_cap






def SLBW_capture(g, k, ac, E, resonance_ladder):
    """
    Calculates a multi-level capture cross section using SLBW formalism.

    _extended_summary_

    Parameters
    ----------
    g : float
        Spin statistical factor $g_{J,\alpha}$.
    k : float or array-like
        Angular wavenumber or array of angular wavenumber values corresponding to the energy vector.
    E : float or array-like
        KE of incident particle or array of KE's.
    resonance_ladder : DataFrame
        DF with columns for 

    Returns
    -------
    xs
        SLBW capture cross section.
    """

    if len(k) != len(E):
        raise ValueError("Vector of angular wavenumbers, k(E), does not match the length of vector E")

    xs = 0
    constant = (np.pi*g/(k**2))
    for index, row in resonance_ladder.iterrows():
        E_lambda = row.E
        # gnx2 = sum([row[ign] for ign in range(2,len(row))]) * 1e-3
        # Gn = 2*(k*ac)*gnx2
        Gn = row.Gn *1e-3
        Gg = row.Gg * 1e-3
        d = (E-E_lambda)**2 + ((Gg+Gn)/2)**2 
        xs += (Gg*Gn) / ( d )
    xs = constant*xs
    return xs
    

