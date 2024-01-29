import numpy as np
from scipy.stats import chi2

from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.theory.scattering_params import FofE_recursive

__doc__ = """
This module contains Bayes' update for the probabilistic distribution on the neutron widths (and
gamma widths if specified).
"""

def Penetrability(E, L, particle_pair:Particle_Pair):
    Lmax = np.max(L)
    P = np.zeros((E.size,L.size))
    for i, l in enumerate(L):
        P[:,i] = FofE_recursive(E, particle_pair.ac, particle_pair.M, particle_pair.m, l)[1]
    return P

def PTBayes(particle_pair:Particle_Pair, lvl_dens_false:float=0.0, false_width_dist=None, prior=None, gamma_width_on:bool=False):
    """
    Performs a Bayesian update on the spingroup probabilities of resonances, based on Porter-Thomas
    Distribution on the neutron widths (and gamma widths if specified).

    Let `L` be the number of resonances and `G` be the number of (true) spingroups.

    Parameters:
    ----------
    particle_pair    :: Particle_Pair
        The particle-pair object holding resonance and mean parameter data.
    lvl_dens_false   :: float
        The level-density for false resonances. Default = 0.0.
    false_width_dist :: function
        The PDF for the neutron widths of false resonances. If none are given, false widths are
        sampled from the joint neutron width PDF of all spingroups. Default is None.
    prior            :: float [L,G+1]
        Optional prior spingroup probabilities. Such probabilities may include information on
        statistical fits. However, the probabilities must be independent of the information
        provided by the width distributions. Default is None.
    gamma_width_on   :: bool
        Determines whether the gamma-width probabilities are calculated based on the theoretical
        distribution. Many RRR evaluations assume the gamma widths are more or less constant. This
        theory is controversial, which is why the gamma width distribution is not considered by
        default. Default is False.

    Returns:
    -------
    posterior        :: float [L,G+1]
        The posterior spingroup probabilities.
    log_likelihood   :: float
        Calculated log-likelihood.
    """

    # Error Checking:
    if not isinstance(particle_pair, Particle_Pair):
        raise TypeError('The "particle_pair" argument must be a "Particle_Pair" object.')
    
    lvl_dens = []; gn2m = []; nDOF = []; gg2m = []; gDOF = []; L = []
    for Jpi, spingroup in particle_pair.spin_groups.items():
        lvl_dens.append(1/spingroup['<D>'])
        gn2m.append(spingroup['<gn2>'])
        nDOF.append(spingroup['n_dof'])
        gg2m.append(spingroup['<gg2>'])
        gDOF.append(spingroup['g_dof'])
        L.append(spingroup['Ls'])
    lvl_dens.append(lvl_dens_false)
    lvl_dens = np.array(lvl_dens)
    gn2m = np.array(gn2m)
    nDOF = np.array(nDOF, dtype=int)
    gg2m = np.array(gg2m)
    gDOF = np.array(gDOF, dtype=int)
    L = np.array(L, dtype=int)

    res = particle_pair.resonance_ladder
    E  = res['E'].to_numpy()
    Gn = res['Gn1'].to_numpy()
    Gg = res['Gg'].to_numpy()
    
    # Setting prior:
    if prior == None:
        prob = lvl_dens / np.sum(lvl_dens)
        prior = np.tile(prob, (res.E.size,1))
    posterior = prior

    # Neutron widths:
    scale = (2*Penetrability(E, L, particle_pair)) * (gn2m/nDOF).reshape(1,-1)
    posterior[:,:-1] *= chi2.pdf(Gn.reshape(-1,1), df=nDOF, scale=scale)

    # Gamma widths: (if gamma_width_on is True)
    if gamma_width_on:
        posterior[:,:-1] *= chi2.pdf(Gg.reshape(-1,1), df=gDOF, scale=(gDOF/(gg2m*2)).reshape(1,-1))

    # False distribution:
    if (lvl_dens_false != 0.0) and (false_width_dist is not None):
        posterior[:,-1] *= false_width_dist(E, Gn, Gg)
    else:
        posterior[:,-1] *= np.sum(posterior[:,:-1], axis=1) / np.sum(prob[:-1])

    # Normalization:
    likelihoods = np.sum(posterior, axis=1)
    posterior /= likelihoods.reshape(-1,1)

    # Log likelihood:
    log_likelihood = np.sum(np.log(likelihoods))

    return posterior, log_likelihood