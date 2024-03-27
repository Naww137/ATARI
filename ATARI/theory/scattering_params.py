#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:54:52 2022

@author: noahwalton
"""

import numpy as np


def gstat(J, I, i):
    """
    Calculates the spin statistical factor for a given $J_{\pi,\alpha}$.

    The spin statistical factor (g) is a weigting factor describing the probability of the different total angular momenta, given by:
    
    $ g_{J\alpha} = \frac{2J+1}{(2i+1)(2I+1)} $ 


    Parameters
    ----------
    J : float or int
        Total angular momentum of the channel.
    I : float or int
        Spin and parity of the target particle.
    i : float or int
        Spin and parity of the incident particle.

    Returns
    -------
    _type_
        _description_

    """
    return (2*J+1)/((2*i+1)*(2*I+1))


def k_wavenumber(E, M, m):
    """
    Calculates the angular wavenumber of the compound state nucleus at a given incident energy, units .

    This function calculates the wavenumber of the compound state nucleus
    created by an incident neutron and a Cu-63 atom. Some nucelar parameters 
    are housed within this function, this could be updated for other nuclei.

    Parameters
    ----------
    E : float or numpy.ndarray
        Energy of the incident neutron.

    Returns
    -------
    float or numpy.ndarray
        Returns either the scalar k or vector of k at different energies.

    """
    # constants 
    hbar = 6.582119569e-16 # eV-s
    c = 2.99792458e8 # m/s
    mn_eV = 939.565420e6 # eV/c^2
    constant = (np.sqrt(2*mn_eV)/c/hbar)*(1e-14) # 0.002197 #sqrt(2Mn)/hbar 

    k = (M/(M+m))*constant*np.sqrt(E)
    return k # 1/√barns 
    

def FofE_recursive(E, ac, M, m, orbital_angular_momentum):
        """
        Calculates functions of energy using recursion.

        This function calculates the centifugal barrier penetrability,
        the shift factor, and the potential scattering phase shift. The recursive implementation
        will allow for high l values. 

        Parameters
        ----------
        E : numpy.ndarray
            Energy of the incident neutron.
        ac : float
            Scattering channel radius in √barns or 1e-12 cm.
        M : float or int
            Mass of the target nucleus.
        m : float or int
            Mass of the incident particle.
        orbital_angular_momentum : int
            Maximum orbital angular momentum of the pair (describes waveform l) to be considered.

        Returns
        -------
        S_array : numpy.ndarray
            Array shift factors, each row is an l value, columns traverse energy vector given.
        P_array : numpy.ndarray
            Array penetrability, each row is an l value, columns traverse energy vector given.
        arcphi_array : numpy.ndarray
            Array of scattering phase shifts, each row is an l value, columns traverse energy vector given.

        See Also
        --------
        FofE_explicit : Calculates functions of energy using explicit definitions, limited to l=3.
        
        Examples
        --------
        >>> from theory.scattering_params import FofE_recursive
        >>> FofE_recursive(np.array([10.4]), 0.786, 180.948030, 1, 2)
        (array([[ 0.        ],
                [-0.99996933],
                [-1.99998978]]),
        array([[5.53780625e-03],
                [1.69824347e-07],
                [5.78684482e-13]]),
        array([[5.53780625e-03],
                [5.66088100e-08],
                [1.15736946e-13]]))
        """
        k = k_wavenumber(E, M, m)
        rho = k*ac

        S_array = np.zeros([int(orbital_angular_momentum+1),len(E)])
        P_array = np.ones([int(orbital_angular_momentum+1),len(E)])
        arcphi_array = np.ones([int(orbital_angular_momentum+1),len(E)])
        P_array[0,:] *= rho
        arcphi_array[0,:] *= rho

        for l in range(1,int(orbital_angular_momentum+1)):
            S = (rho**2*(l-S_array[l-1]) / ((l-S_array[l-1])**2 + P_array[l-1]**2)) - l
            P = rho**2*P_array[l-1] / ((l-S_array[l-1])**2 + P_array[l-1]**2)
            arcphi = arcphi_array[l-1] - np.arctan(P_array[l-1]/(l-S_array[l-1]))
            
            S_array[l,:] = S
            P_array[l,:] = P
            arcphi_array[l,:] = arcphi
            
        return S_array, P_array, arcphi_array, k



def FofE_explicit(E, ac, M, m, orbital_angular_momentum):
    """
    Calculates penetrability and shift functions using explicit definitions.

    This function calculates the centifugal barrier penetrability as well as
    the shift factor for a neutron incident on Cu-63. The explicit implementation
    only allows for l-values up to 3. Some nucelar parameters are housed within 
    this function, this could be updated for other nuclei.

    Parameters
    ----------
    E : float or numpy.ndarray
        Energy of the incident neutron.
    ac : float
        Scattering channel radius in √barns or 1e-12 cm.
    M : float or int
            Mass of the target nucleus.
    m : float or int
        Mass of the incident particle.
    orbital_angular_momentum : int
        Orbital angular momentum of the pair, describes waveform (l).

    Returns
    -------
    S_array : array-like
        Shift factor(s) at given energy.
    P_array : array-like
        Penetrability at given energy.
    psi : array-like
        Potential scattering phase shift.

    See Also
    --------
    PS_recursive : Calculates functions of energy using recursion.
    
    """
    
    assert(orbital_angular_momentum == 0)# "Phase shift function in syndat.scattering theory needs to be updated for higher-order waveforms"

    k = k_wavenumber(E, M, m)
    rho = k*ac
    phi = rho
    
    if orbital_angular_momentum == 0:
        P = rho
        S = np.zeros(len(E))
    elif orbital_angular_momentum == 1:
        P = rho**3/(1+rho**2)
        S = -1/(1+rho**2)
    elif orbital_angular_momentum == 2:
        P = rho**5/(9+3*rho**2+rho**4)
        S = -(18+3*rho**2)/(9+3*rho**2+rho**4)
    elif orbital_angular_momentum == 3:
        P = rho**7/(255+45*rho**2+6*rho**4+rho**6)
        S = -(675+90*rho**2+6*rho**4)/(225+45*rho**2+6*rho**4+rho**6)

    else:
        raise ValueError("PS_explicit cannot handle orbital_angular_momenta > 3, use PS_recursive")
        
    return S, P, phi, k


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


# NOTE: Duplicate, unused:
def reduced_width_square_2_partial_width(E, ac, M, m, reduced_widths_square, orbital_angular_momentum):
    S, P, psi, k = FofE_recursive(np.array(E), ac, M, m, orbital_angular_momentum)
    partial_widths = 2*P*reduced_widths_square 
    return partial_widths
