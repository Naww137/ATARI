from typing import Tuple, List
import numpy as np
from pandas import Series

from TAZ.Encore import Encore
from TAZ.Theory import SpacingDistributionBase, merge

import warnings

__doc__ = """
This module serves as a preprocessor and postprocessor for the spingroup assignment algorithm,
"Encore". The "merge" feature can compress an assignment problem with more than 2 spingroups into a
multiple 2-spingroup assignment problems. The merge feature may be needed when the number of
spingroups is 4 or higher as the problem would become computationally infeasible. The merge
feature finds the probabilities for various merge cases and combines the expected probabilities.
"""

class RunMaster:
    f"""
    A wrapper over Encore responsible for calculating level-spacing probabilities, and providing
    the "merge" functionality. Once a RunMaster object has been initialized, the specific algorithm
    can be chosen (i.e. WigBayes, WigSample, LogLikelihood, etc.). The WigMaxLikelihood does not
    require Encore to be initialized, so it can be called as a class method without initialization.

    Methods
    -------
    WigBayes
        Calculates the probability of each spingroup assignment for each resonance.
    WigMaxScore
        Returns the spingroup assignments that maximize the number of correct spingroup
        assignments.
    WigSample
        Samples an ensemble of spingroup assignments.
    LogLikelihood
        The logarithm of the likelihood of sampling the resonance distribution assuming the
        provided mean parameters.
    ProbOfSample
        Returns the likelihood of observing the provided spingroup assignment given the resonance
        ladder.
    WigMaxLikelihood
        Returns the maximum likelihood spingroup assignments using branching and pruning methods.
    """

    # =============================================================================================
    # Initialization and Creating Pipeline
    # =============================================================================================

    def __init__(self, E, energy_range:tuple,
                 level_spacing_dists:Tuple[SpacingDistributionBase], false_dens:float=0.0,
                 prior=None, log_likelihood_prior:float=None,
                 err:float=1e-9, merge:bool=False,
                 verbose:bool=False):
        """
        Initializes a RunMaster object.

        Parameters
        ----------
        E                    : array-like of float
            Resonance energies for the ladder.
        energy_range         : (float, float)
            The ladder energy boundaries.
        level_spacing_dists  : ndarray[SpacingDistribution]
            The level-spacing distributions object.
        false_dens           : float
            The false level-density. Default = 0.0.
        prior                : array-like of float, optional
            The prior probabilitiy distribution for each spingroup.
        log_likelihood_prior : float, optional
            The log-likelihood provided from the prior.
        err                  : float
            A level-spacing probability threshold at which resonances are considered to be too far
            apart to be nearest neighbors.
        merge                : bool
            Merges level-spacings if True. Default = False.
        verbose              : bool
            The verbosity controller. Default = False.
        """

        # Error Checking:
        if not (0.0 < err < 1.0):
            raise ValueError('The probability threshold, "err", must be strictly between 0 and 1.')
        if len(energy_range) != 2:
            raise ValueError('"energy_range" must be a tuple with two elements: an lower and upper bound on the resonance ladder energies.')
        if energy_range[0] >= energy_range[1]:
            raise ValueError('energy_range[0] must be strictly less than energy_range[1].')
        
        if isinstance(E, Series):
            E = E.to_numpy()
        self.E  = np.sort(E)
        self.energy_range = tuple(energy_range)
        self.level_spacing_dists = np.array(level_spacing_dists)
        self.lvl_dens = np.array([lvl_spacing_dist.lvl_dens for lvl_spacing_dist in level_spacing_dists] + [false_dens])

        self.L = len(E) # Number of resonances
        self.G = len(self.lvl_dens) - 1 # number of spingroups (not including false group)

        if prior is None:
            self.prior = np.tile(self.lvl_dens/self.lvl_dens_tot, (self.L,1))
        else:
            self.prior = prior
        self.log_likelihood_prior = log_likelihood_prior
        self.verbose = verbose

        if merge and self.G <= 2:
            warnings.warn('Merging does not occur for G <= 2. merge will be set to False.', UserWarning)
            merge = False
        self.merge = merge

        self._prepare_encore_pipe(err)

    def _prepare_encore_pipe(self, err:float):
        """
        Prepares the preprocessing for Encore such as level-spacing probability calculations. Once
        the Encore dependences are prepared Encore is initialized for each required runcase.

        Parameters
        ----------
        err : float
            A level-spacing probability threshold at which resonances are considered to be too far
            apart to be nearest neighbors.
        """

        if self.merge:
            self.encore_pipes = []
            for g in range(self.G):
                partition = ([g_ for g_ in range(self.G) if g_ != g], [g])
                encore_pipe = self._partition_encore(partition, err)
                self.encore_pipes.append(encore_pipe)
            # Base partition:
            partition = ([g for g in range(self.G)],)
            encore_pipe = self._partition_encore(partition, err)
            self.encore_pipes.append(encore_pipe)
        else:
            iMax = np.zeros((self.L+2, 2, self.G), 'i4')
            level_spacing_probs = np.zeros((self.L+2, self.L+2, self.G), 'f8')
            for g in range(self.G):
                if self.verbose:    print(f'Finding level-spacing probabilities for group {g}.')
                iMax[:,:,g] = self._calculate_iMax(self.E, self.energy_range, self.level_spacing_dists[g], err)
                level_spacing_probs[:,:,g] = self._calculate_probs(self.E, self.energy_range, self.level_spacing_dists[g], iMax[:,:,g])
            if self.verbose:    print(f'Creating ENCORE pipeline.')
            self.encore_pipe = Encore(self.prior, level_spacing_probs, iMax)
            if self.verbose:    print(f'Finished ENCORE initialization.')

    def _partition_encore(self, partition:Tuple[List[int]], err:float):
        """
        In the case a merge is required, _partition_encore takes a spingroup partition, calculates
        the encore inputs such as level-spacing probabilities, and returns an Encore object.

        Parameters
        ----------
        partition : Tuple[List[int]]
            A partition of the provided spingroup identifiers.
        err       : float
            A level-spacing probability threshold at which resonances are considered to be too far
            apart to be nearest neighbors.

        Returns
        -------
        encore_pipe : Encore
            An Encore object for the given spingroup partition.
        """

        num_partitions = len(partition)
        level_spacing_probs = np.zeros((self.L+2, self.L+2, num_partitions), 'f8')
        iMax = np.zeros((self.L+2, 2, num_partitions), 'i4')
        prior_merged = np.zeros((self.L, num_partitions+1), 'f8')
        for g, groups in enumerate(partition):
            if self.verbose:    print(f'Partitioning groups {groups} for partition {g}.')
            distribution = merge(*self.level_spacing_dists[groups])
            if self.verbose:    print(f'Finding level-spacing probabilities for groups {groups}.')
            iMax[:,:,g] = self._calculate_iMax(self.E, self.energy_range, distribution, err)
            level_spacing_probs[:,:,g] = self._calculate_probs(self.E, self.energy_range, distribution, iMax[:,:,g], self.prior[:,groups])
            if hasattr(groups, '__iter__'):
                prior_merged[:,g] = np.sum(self.prior[:,groups], axis=1)
            else:
                prior_merged[:,g] = self.prior[:,groups]
        prior_merged[:,-1] = self.prior[:,-1]
        if self.verbose:    print(f'Creating ENCORE pipeline for group {g}.')
        encore_pipe = Encore(prior_merged, level_spacing_probs, iMax)
        if self.verbose:    print(f'Finished ENCORE initialization for partition, {partition}.')
        return encore_pipe
        
    @staticmethod
    def _calculate_iMax(E, energy_range:tuple,
                        distribution:SpacingDistributionBase, err:float):
        """
        Calculates the index limits for calculating level-spacings. 

        Parameters
        ----------
        E            : ndarray[float]
            The ordered resonance energies.
        energy_range : (float, float)
            A tuple providing the lower and upper energies for the grid.
        distribution : SpacingDistribution
            The SpacingDistribution object for the spingroup or merged group.
        err          : float
            A level-spacing probability threshold at which resonances are considered to be too far
            apart to be nearest neighbors.

        Returns
        -------
        iMax : int32[L+2,2]
            An array of the lower and upper limits for the resonance index of the nearest-
            neighboring resonance to the resonance index given by the row index. If the second
            index is 0, the lower limit is provided. If the second index is 1, the upper limit is
            provided.
        """

        L = E.size
        iMax = np.full((L+2,2), -1, dtype='i4')
        xMax_f0 = distribution.xMax_f0(err)
        if xMax_f0 <= 0.0:          raise RuntimeError('xMax_f0 was found to be negative.')
        elif np.isinf(xMax_f0):     raise RuntimeError('xMax_f0 was found to be infinite.')
        elif np.isnan(xMax_f0):     raise RuntimeError('xMax_f0 was found to be NaN.')
        xMax_f1 = distribution.xMax_f1(err)
        if xMax_f1 <= 0.0:          raise RuntimeError('xMax_f1 was found to be negative.')
        elif np.isinf(xMax_f1):     raise RuntimeError('xMax_f1 was found to be infinite.')
        elif np.isnan(xMax_f1):     raise RuntimeError('xMax_f1 was found to be NaN.')

        # Lower boundary cases:
        for j in range(L):
            if E[j] - energy_range[0] >= xMax_f1:
                iMax[0,0]    = j
                iMax[:j+1,1] = 0
                break

        # Intermediate cases:
        for i in range(L-1):
            for j in range(iMax[i,0]+1,L):
                if E[j] - E[i] >= xMax_f0:
                    iMax[i+1,0] = j
                    iMax[iMax[i-1,0]:j+1,1] = i+1
                    break
            else:
                iMax[i:,0] = L+1
                iMax[iMax[i-1,0]:,1] = i+1
                break

        # Upper boundary cases:
        for j in range(L-1,-1,-1):
            if energy_range[1] - E[j] >= xMax_f1:
                iMax[-1,1] = j
                iMax[j:,0] = L+1
                break

        return iMax
    
    @staticmethod
    def _calculate_probs(E, energy_range:tuple,
                         distribution:SpacingDistributionBase, iMax,
                         prior=None):
        """
        Calculates the level-spacing probabilities using the provided distribution and stored
        resonance energies. 

        Parameters
        ----------
        E            : ndarray[float]
            The ordered resonance energies.
        energy_range : (float, float)
            A tuple providing the lower and upper energies for the grid.
        distribution : SpacingDistribution
            The SpacingDistribution object for the spingroup or merged group.
        iMax         : int32[L+2,2]
            An array of the lower and upper limits for the resonance index of the nearest-
            neighboring resonance to the resonance index given by the row index. If the column
            index is 0, the lower limit is provided. If the column index is 1, the upper limit is
            provided.
        prior        : float[L+2,S]
            The prior spingroup probabilities for each resonance, given by the row index, for each
            spingroup in the merged group, given by the column index. Default is None.

        Returns
        -------
        level_spacing_probs : float64[L+2,L+2]
            The level-spacing probabilities for the group where the first and second index
            represent the resonance identifiers at which the level-spacing probabilites are
            between.
        """

        L = E.size
        level_spacing_probs = np.zeros((L+2,L+2), dtype='f8')
        for i in range(L-1):
            X = E[i+1:iMax[i+1,0]-1] - E[i]
            if prior is None:
                lvl_spacing_prob = distribution.f0(X)
            else:
                prior_L = np.tile(prior[i,:], (iMax[i+1,0]-i-2, 1))
                prior_R = prior[i+1:iMax[i+1,0]-1,:]
                lvl_spacing_prob = distribution.f0(X, prior_L, prior_R)
            level_spacing_probs[i+1,i+2:iMax[i+1,0]] = lvl_spacing_prob
        # Boundary distribution:
        if prior is None:
            level_spacing_probs[0,1:-1]  = distribution.f1(E - energy_range[0])
            level_spacing_probs[1:-1,-1] = distribution.f1(energy_range[1] - E)
        else:
            level_spacing_probs[0,1:-1]  = distribution.f1(E - energy_range[0], prior)
            level_spacing_probs[1:-1,-1] = distribution.f1(energy_range[1] - E, prior)

        # Error checking:
        if (level_spacing_probs == np.nan).any():   raise RuntimeError('Level-spacing probabilities have "NaN" values.')
        if (level_spacing_probs == np.inf).any():   raise RuntimeError('Level-spacing probabilities have "Inf" values.')
        if (level_spacing_probs <  0.0).any():      raise RuntimeError('Level-spacing probabilities have negative values.')

        # The normalization factor is duplicated in the prior. One must be removed: FIXME!!!!!
        level_spacing_probs /= distribution.lvl_dens
        return level_spacing_probs
    
    # =============================================================================================
    # Combining Merge Case Probabilities
    # =============================================================================================
    
    def _prob_combinator(self, sg_probs):
        """
        Combines probabilities from various spingroup partitions.

        ...
        """

        combined_sg_probs = np.zeros((self.L,self.G+1), dtype='f8')
        
        combined_sg_probs[:,:-1] = sg_probs[:,1,:] # lone spingroup
        
        combined_sg_probs[:,-1] = np.prod(sg_probs[:,2,:], axis=1) * self.prior[:,-1] ** (1-self.G)
        combined_sg_probs[self.prior[:,-1]==0.0, -1] = 0.0
        
        combined_sg_probs /= np.sum(combined_sg_probs, axis=1, keepdims=True)
        return combined_sg_probs

    def _log_likelihood_combinator(self, partition_log_likelihoods, base_log_likelihoods:float):
        """
        Combines log-likelihoods from from various partitions.
        """

        # FIXME: I DON'T KNOW LOG LIKELIHOOD CORRECTION FACTOR FOR MERGED CASES! 
        return np.sum(partition_log_likelihoods) - (self.G-1)*base_log_likelihoods
    
    # =============================================================================================
    # Properties
    # =============================================================================================

    @property
    def lvl_dens_tot(self):
        'Total level-density over all spingroups.'
        return np.sum(self.lvl_dens)
    @property
    def false_dens(self):
        'Level-density for false resonances.'
        return self.lvl_dens[-1]
    
    # =============================================================================================
    # Methods
    # =============================================================================================

    def WigBayes(self):
        """
        Returns spingroup probabilities for each resonance based on level-spacing distributions,
        and any provided prior.

        Returns
        -------
        sg_probs : int [L,G]
            The spingroup probabilities for each resonance.
        """

        if self.merge:
            sg_probs = np.zeros((self.L,3,self.G),dtype='f8')
            for g, encore_pipe in enumerate(self.encore_pipes[:-1]):
                if self.verbose:    print(f'Starting WigBayes for partition {g}')
                sg_probs[:,:,g] = encore_pipe.WigBayes()
                if self.verbose:    print(f'Finished running WigBayes for partition {g}')
            print(f'Combining Probabilities')
            combined_sg_probs = self._prob_combinator(sg_probs)
            print(f'Finished calculation')
            return combined_sg_probs
        else:
            if self.verbose:    print('Starting WigBayes')
            sg_probs = self.encore_pipe.WigBayes()
            if self.verbose:    print('Finished running WigBayes')
            return sg_probs
        
    def WigMaxScore(self):
        """
        Returns the spingroup assignments that maximize the number of correct spingroup
        assignments. This is the same as taking the maximum probable spingroup for each resonance
        from the WigBayes probabilities.

        Returns
        -------
        spingroups : int [L]
            The spingroup probabilities for each resonance.
        """

        sg_probs = self.WigBayes()
        spingroups = np.max(sg_probs, axis=1)
        return spingroups
    
    def WigSample(self, num_trials:int=1, rng:np.random.Generator=None, seed:int=None):
        """
        Returns random spingroup assignment samples based on its Bayesian probability.

        Parameters
        ----------
        num_trials : int
            Determines the number of sampled ensembles to return. Default is 1.
        rng        : np.random.Generator
            The random number generator for random sampling. Default is None.
        seed       : int
            The random number seed for random sampling. Default is None.

        Returns
        -------
        samples : int [L,trials]
            The sampled IDs for each resonance and trial.
        """

        if self.merge:
            raise NotImplementedError('WigSample for the merged case has not been implemented yet.')
        else:
            if self.verbose:    print('Starting WigSample')
            samples = self.encore_pipe.WigSample(num_trials, rng=rng, seed=seed)
            if self.verbose:    print('Finished running WigSample')
            return samples
    
    def LogLikelihood(self):
        """
        Returns the log-likelihoods for the resonance parameters, regardless of spingroup
        assignment.

        Returns
        -------
        log_likelihood : float
            The logarithm of the likelihood of sampling the resonance ladder provided the mean
            parameters.
        """

        if self.merge:
            log_likelihoods = np.zeros(self.G, dtype='f8')
            for g, encore_pipe in enumerate(self.encore_pipes[:-1]):
                log_likelihoods[g] = encore_pipe.LogLikelihood(self.energy_range, self.false_dens, self.log_likelihood_prior) # FIXME this may have incorrect prior
            base_log_likelihood = self.encore_pipes[-1].LogLikelihood(self.energy_range, self.false_dens, self.log_likelihood_prior)
            combined_log_likelihood = self._log_likelihood_combinator(log_likelihoods, base_log_likelihood)
            return combined_log_likelihood
        else:
            log_likelihood = self.encore_pipe.LogLikelihood(self.energy_range, self.false_dens, self.log_likelihood_prior)
            return log_likelihood
        
    def ProbOfSample(self, spingroup_assignments):
        """
        Returns the likelihood of observing the provided spingroup assignment given the resonance
        ladder.

        Parameters
        ----------
        spingroup_assignments : array-like of int
            A list of spingroup assignment IDs for each resonance.

        Returns
        -------
        likelihood : float
            The likelihood.
        """

        # NOTE: This needs checking!!!
        if self.merge:
            raise NotImplementedError('ProbOfSample for the merged case has not been implemented yet.')
        else:
            likelihood = self.encore_pipe.ProbOfSample(spingroup_assignments)
        return likelihood
    
    @classmethod
    def WigMaxLikelihoods(cls, E, energy_range:tuple,
                         level_spacing_dists:Tuple[SpacingDistributionBase], false_dens:float=0.0,
                         num_best:int=None,
                         err:float=1e-8,
                         prior=None):
        """
        Returns the maximum likelihood spingroup assignments using branching and pruning methods.
        This method does not need the CP values from initialization. Therefore, WigMaxLikelihood
        is a class method that can be called prior to initialization.

        Parameters
        ----------
        E                   : array-like of float
            The ordered resonance energies.
        energy_range        : (float, float)
            The lower and upper energy limits of the resonance ladder.
        level_spacing_dists : Tuple[SpacingDistributionBase]
            The level-spacing distributions for each spingroup (not including false group).
        false_dens          : float
            The level-density for the false group. Default = 0.0.
        num_best            : int, optional
            The number of highest likelihood spingroup ladders to select. By default,
            provides the highest likelihood ladder only.
        err                 : float
            The maximum error allowed, beyond which level-spacings are not calculated.
            Default is 1e-8.
        prior               : array-like of float, optional
            The prior spingroup probabilities for each resonance.

        Returns
        -------
        best_spingroup_ladders : list[ndarray[int]]
            List of the best spingroup ladders, stored as arrays of spingroup IDs.
        best_log_likelihoods   : list[float]
            Log-likelihoods for the best spingroup assignments.
        """
        
        L = len(E)
        G = len(level_spacing_dists)

        if isinstance(E, Series):
            E = E.to_numpy()
        E = np.sort(E)

        if prior is None:
            prior = np.zeros((L,G+1))
            for g, distribution in enumerate(level_spacing_dists):
                prior[:,g] = distribution.lvl_dens
            prior[:,G] = false_dens
            prior /= np.sum(prior, axis=1)

        iMax = np.zeros((L+2, 2, G), 'i4')
        level_spacing_probs = np.zeros((L+2, L+2, G), 'f8')
        for g, distribution in enumerate(level_spacing_dists):
            iMax[:,:,g] = cls._calculate_iMax(E, energy_range, distribution, err)
            level_spacing_probs[:,:,g] = cls._calculate_probs(E, energy_range, distribution, iMax[:,:,g])
        if num_best is None:
            best_spingroup_ladders, best_log_likelihoods = Encore.WigMaxLikelihoods(prior, level_spacing_probs, iMax, 1)
            best_spingroup_ladder = best_spingroup_ladders[0]
            best_log_likelihood   = best_log_likelihoods  [0]
            return best_spingroup_ladder, best_log_likelihood
        else:
            best_spingroup_ladders, best_log_likelihoods = Encore.WigMaxLikelihoods(prior, level_spacing_probs, iMax, num_best)
            return best_spingroup_ladders, best_log_likelihoods