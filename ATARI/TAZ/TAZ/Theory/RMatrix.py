import numpy as np

from ATARI.ModelData.particle import mass_neutron

__doc__ = """
This module is the collection of relevant R-Matrix Theory quantities. Many of these equations are
found in the ENDF and SAMMY manuals.
"""

# Physical constants: (Table 1 in appendix H of ENDF manual)
HBAR       = 6.582_119_514e-16 # eV*s
LIGHTSPEED = 299_792_458 # m/s
AMU_EV     = 931.494_095_4e6 # eV/(c^2*amu)

def k_wavenumber(mass_targ:float, E,
                 mass_proj:float=mass_neutron,
                 mass_targ_after:float=None,
                 mass_proj_after:float=None,
                 E_thres:float=None):
    """
    Finds the k wavenumber.

    Based on equation II A.9 in the SAMMY manual.

    Parameters
    ----------
    mass_targ       : float
        Mass of the target isotope.
    E               : float, array-like
        Energy points for which Rho is evaluated.
    mass_proj       : float
        Mass of the projectile. Default = 1.008665 amu (neutron mass).
    mass_targ_after : float
        Mass of the target after the reaction. Default = mass_targ.
    mass_proj_after : float
        Mass of the target before the reaction. Default = mass_proj.
    E_thres         : float
        Threshold energy for the reaction. Default is calculated from Q-value.
        
    Returns
    --------
    k : float, array-like
        Momentum factor, ρ.
    """

    E = np.array(E)
    if mass_targ_after is None:
        mass_targ_after = mass_targ # assume elastic scattering
    if mass_proj_after is None:
        mass_proj_after = mass_proj # assume elastic scattering
    if E_thres is None:
        Q_value = mass_targ + mass_proj - mass_targ_after - mass_proj_after
        E_thres = - ((mass_targ + mass_proj)/mass_targ) * Q_value # Eq. II C2.1 in the SAMMY manual

    # Error Checking:
    if any(E < E_thres):
        raise ValueError(f'The given energies are below the threshold energy of {E_thres} eV.')

    CONSTANT = np.sqrt(AMU_EV / LIGHTSPEED**2) / HBAR * 1e-14 # = 0.0001546691274 -- (√amu * √b) / h_bar --> √eV

    mass_ratio_before = mass_targ / (mass_proj + mass_targ)
    mass_ratio_after  = 2 * mass_proj_after * mass_targ_after / (mass_proj_after + mass_targ_after)
    Delta_E = E-E_thres
    k = CONSTANT * np.sqrt((mass_ratio_before * mass_ratio_after) * Delta_E)
    return k

def rho(k, ac:float):
    """
    Finds the momentum factor, ρ.

    Based on equation II A.9 in the SAMMY manual.

    Parameters
    ----------
    k  : float, array-like
        The k-wavenumber(s).
    ac : float
        Channel radius.
        
    Returns
    --------
    rho : float, array-like
        Momentum factor, ρ.
    """

    rho = k * ac
    return rho

def penetration_factor(rho, l:int):
    """
    Finds the Penetration factor.

    Based on table II A.1 in the SAMMY manual.

    Parameters
    ----------
    rho : float, array-like
        Momentum factor.
    l   : int, array-like
        Orbital angular momentum quantum number.

    Returns
    --------
    pen_factor : float, array-like
        Penetration factor.
    """

    def _penetration_factor(rho, l:int):
        rho2 = rho**2
        if   l == 0:
            return rho
        elif l == 1:
            return rho*rho2    / (  1 +    rho2)
        elif l == 2:
            return rho*rho2**2 / (  9 +  3*rho2 +   rho2**2)
        elif l == 3:
            return rho*rho2**3 / (225 + 45*rho2 + 6*rho2**2 + rho2**3)
        else: # l >= 4
            
            # l = 3:
            denom = (225 + 45*rho2 + 6*rho2**2 + rho2**3)
            P = rho*rho2**3 / denom
            S = -(675 + 90*rho2 + 6*rho2**2) / denom

            # Iteration equation:
            for l_iter in range(4,l+1):
                mult = rho2 / ((l_iter-S)**2 + P**2)
                P = mult*P
                S = mult*S - l_iter
            return P

    if hasattr(l, '__iter__'): # is iterable
        pen_factor = np.zeros((len(rho),len(l)))
        for g, lg in enumerate(l):
            pen_factor[:,g] = _penetration_factor(rho,lg)
    else: # is not iterable
        pen_factor = np.array(_penetration_factor(rho,l))
    return pen_factor

def G_to_g2(G, penetrability):
    """
    Converts partial widths to reduced widths.

    Parameters
    ----------
    G : float, array-like
        The partial width.
    penetrability : float, array-like
        The penetration factor.

    Returns
    -------
    g2 : float, array-like
        The reduced width.
    """

    g2 = G / (2 * penetrability)
    return g2

def g2_to_G(g2, penetrability):
    """
    Converts reduced widths to partial widths.

    Parameters
    ----------
    g2 : float, array-like
        The reduced width.
    penetrability : float, array-like
        The penetration factor.

    Returns
    -------
    G : float, array-like
        The partial width.
    """

    G = 2 * penetrability * g2
    return G

def g_to_g2(g):
    """
    Converts reduced width amplitude(s) to reduced width(s).

    Parameters
    ----------
    g : float, array-like
        Reduce width amplitude(s).

    Returns
    -------
    g2 : float, array-like
        Reduced width(s).
    """

    g2 = g * abs(g)
    return g2

def g2_to_g(g2):
    """
    Converts reduced width(s) to reduced width amplitude(s).

    Parameters
    ----------
    g2 : float, array-like
        Reduce width(s).

    Returns
    -------
    g : float, array-like
        Reduced width amplitude(s).
    """

    g = np.sign(g2) * np.sqrt(abs(g2))
    return g