import numpy as np
from scipy.integrate import quad as integrate
from scipy.optimize import root_scalar

from ATARI.TAZ.Encore import Encore
from ATARI.TAZ.distributions import Distributions

__doc__ = """
This module serves as a preprocessor and postprocessor for the 2-spingroup assignment algorithm,
"Encore.py". This module extends the 2-spingroup algorithm to multiple-spingroups by "merging"
spingroups. This module finds the probabilities for various merge cases and combines the expected
probabilities. Unlike the 2-spingroup case that gives the best answer given its information
(i.e. mean level spacings, reduced widths, false resonance probabilities, etc.), the
multiple-spingroup case is an approximation.
"""

# =================================================================================================
#     Merger:
# =================================================================================================
class Merger:
    """
    This class provides the level-spacing distribution of spingroups that have been merged.

    Initialization Attributes:
    -------------------------
    level_spacing_dists :: Distributions
        The level-spacing distribution(s) for each spingroup to calculate wigner probabilities
        with.
    err                 :: float
        Threshold for no longer calculating Wigner probabilities for larger level-spacings.
        Default = 1e-9.
        
    Internally Used Attributes:
    --------------------------
    G        :: int
        The number of possible (true) spingroups.
    lvl_dens :: float [G]
        Level densities for each spingroup. Inverse of mean level spacing.
    Z        :: float [G,G]
        Matrix used to calculate the normalization factor for the merged group level-spacing
        distribution. See the "Z" variable in the merged level-spacing equation.
    ZB       :: float [G]
        A special variation of `Z` used for the level-spacing with one resonance that exists
        outside the ladder energy bounds. As such the resonance's position is assumed to be
        unknown except that we know it is not inside the ladder energy bounds.
    xMax     :: float
        The maximum level spacings that give a probability above a threshold error, `err`. No
        level-spacings are calculated 
    xMaxB    :: float
        A special variation of `xMax` used for the level-spacing with one resonance that exists
        outside the ladder energy bounds. As such the resonance's position is assumed to be
        unknown except that we know it is not inside the ladder energy bounds.
    """

    def __init__(self, level_spacing_dists:Distributions, err:float=1e-9):
        """
        Creates a Merger object from the level-spacing distributions of all (true) spingroups.

        Parameters:
        ----------
        level_spacing_dists :: Distributions
            The level-spacing distributions of all (true) spingroups as provided by the
            level-spacing distribution class in `RMatrix.py`.
        err                 :: float
            A probability threshold where resonances are considered to be too far apart to be
            nearest neighbors.
        """

        # Error Checking:
        if type(level_spacing_dists) != Distributions:
            raise TypeError('The level-spacing distributions must be a "Distributions" object.')
        if not (0.0 < err < 1.0):
            raise ValueError('The probability threshold, "err", must be strictly between 0 and 1.')

        self.level_spacing_dists = level_spacing_dists
        self.lvl_dens = self.level_spacing_dists.lvl_dens
        self.G    = len(self.lvl_dens)

        xMax_limits = self.__xMax_limits(err) # xMax must be bounded by the error for spingroup alone

        if self.G != 1:
            self.Z, self.ZB = self.__findZ(xMax_limits)

        self.xMax, self.xMaxB = self.__findxMax(err, xMax_limits)

    @property
    def lvl_dens_tot(self):
        'Gives the total level density for the combined group.'
        return np.sum(self.lvl_dens)
    @property
    def MLSTot(self):
        'Gives the combined mean level-spacing for the group.'
        return 1.0 / self.lvl_dens_tot

    def __xMax_limits(self, err:float):
        """
        In order to estimate `xMax` and `xMaxB`, we first must calculate `Z` and `ZB`. `xMax_limit`
        is used to get a simple upper bound on `xMax` and `xMaxB`. This limit assumes that the
        `xMax` for the level-spacing distribution is at least less than the `xMax` for Poisson
        distribution.
        """
        xMax = -np.log(err)/self.lvl_dens
        return xMax

    def __findZ(self, xMax_limits):
        """
        A function that calculates normalization matrices using the spingroups. When paired with
        the prior probabilities, the arrays, `Z` and `ZB` are the normalization factors for the
        merged level-spacing probability density function.
        """

        def offDiag(x, i:int, j:int):
            F2, R1, R2 = self.level_spacing_dists.parts(x)
            C = np.prod(F2, axis=1)
            return C[0] * R2[0,i] * R2[0,j]
        def mainDiag(x, i:int):
            F2, R1, R2 = self.level_spacing_dists.parts(x)
            C = np.prod(F2, axis=1)
            return C[0] * R1[0,i] * R2[0,i]
        def boundaries(x, i:int):
            F2, R1, R2 = self.level_spacing_dists.parts(x)
            C = np.prod(F2, axis=1)
            return C[0] * R2[0,i]

        # Level Spacing Normalization Matrix:
        Z = np.zeros((self.G,self.G), dtype='f8')
        for i in range(self.G):
            for j in range(i):
                # Off-diagonal:
                min_xMax_limit = min(xMax_limits[i], xMax_limits[j])
                Z[i,j] = integrate(lambda _x: offDiag(_x,i,j), a=0.0, b=min_xMax_limit)[0]
                Z[j,i] = Z[i,j]
            # Main diagonal:
            Z[i,i] = integrate(lambda _x: mainDiag(_x,i), a=0.0, b=xMax_limits[i])[0]
        
        # Level Spacing Normalization at Boundaries:
        ZB = np.array([integrate(lambda _x: boundaries(_x,i), a=0.0, b=xMax_limits[i])[0] for i in range(self.G)], dtype='f8')

        # Error Checking:
        if   (Z  == np.nan).any():  raise RuntimeError('The normalization matrix, "Z", has some "NaN" values.')
        elif (Z  == np.inf).any():  raise RuntimeError('The normalization matrix, "Z", has some "Inf" values.')
        elif (Z  <  0.0   ).any():  raise RuntimeError('The normalization matrix, "Z", has some negative values.')
        elif (ZB == np.nan).any():  raise RuntimeError('The normalization array, "ZB", has some "NaN" values.')
        elif (ZB == np.inf).any():  raise RuntimeError('The normalization array, "ZB", has some "Inf" values.')
        elif (ZB <  0.0   ).any():  raise RuntimeError('The normalization array, "ZB", has some negative values.')
        return Z, ZB
    
    def __findxMax(self, err:float, xMax_limits):
        """
        An upper limit on the level-spacing beyond which is improbable, as determined by the
        probability threshold, `err`. To ensure the solution is found a rough estimate must be
        provided: `xMax_limit`.
        """
        mthd   = 'brentq' # method for root-finding
        bounds = [self.MLSTot, np.max(xMax_limits)]
        if self.G == 1:
            xMax  = root_scalar(lambda x: self.level_spacing_dists.f0(x) - err, method=mthd, bracket=bounds).root
            xMaxB = root_scalar(lambda x: self.level_spacing_dists.f1(x) - err, method=mthd, bracket=bounds).root
        else:
            # Bounding the equation from above:
            def upperBoundLevelSpacing(x):
                f2, r1, r2 = self.level_spacing_dists.parts(x)
                c = np.prod(f2, axis=1)
                # Minimum normalization possible. This is the lower bound on the denominator with regards to the priors:
                Norm_LB = np.min(self.Z)
                # Finding maximum numerator for the upper bound on the numerator:
                case_1 = np.max(r1*r2, axis=1)
                case_2 = np.max(r2**2)
                return (c / Norm_LB) * max(case_1, case_2)
            def upperBoundLevelSpacingBoundary(x):
                f2, r1, r2 = self.level_spacing_dists.parts(x)
                c = np.prod(f2, axis=1)
                # Minimum normalization possible for lower bound:
                Norm_LB = np.min(self.ZB)
                return (c / Norm_LB) * np.max(r2, axis=1) # Bounded above by bounding the priors from above
            
            xMax  = root_scalar(lambda x: upperBoundLevelSpacing(x)-err        , method=mthd, bracket=bounds).root
            xMaxB = root_scalar(lambda x: upperBoundLevelSpacingBoundary(x)-err, method=mthd, bracket=bounds).root
        return xMax, xMaxB

    def findLevelSpacings(self, E, EB:tuple, prior, verbose:bool=False):
        """
        Finds the level-spacing probabilities, `level_spacing_probs` between each resonance,
        including resonance outside the left and right ladder energy bounds. It is unnecessary
        to calculate all `(L+2)**2` possibilities since the probabilities from most level-spacings
        would be negligible, so we calculate the left and right cutoffs using `xMax` and `xMaxB`
        and store the left and right bounds in `iMax`. `level_spacing_probs` stores all of the
        level-spacing probabilities between those bounds.

        Let `L` be the number of resonances and `G` be the number of spingroups.

        Parameters:
        ----------
        E       :: float [L]
            All recorded resonance energies within the selected energy range, provided by `EB`.
        EB      :: float [2]
            The lower and upper boundaries on the recorded energy range.
        prior   :: float [L,G]
            The prior spingroup probabilities for each resonance and possible (true) spingroup.
        verbose :: bool
            The verbosity controller. Default = False.

        Returns:
        -------
        level_spacing_probs :: float [L+2,L+2]
            The level-spacing probabilities between each resonance, given by the provided index,
            including resonances beyond the lower boundary (first index = 0) and resonances
            beyond the upper boundary (second index = L+1).
        iMax                :: int [L+2,2]
            An array of indices representing the outermost resonances where the level-spacing is
            not negligible as determined by `xMax` and `xMaxB`. `iMax[i,0]` is returns the lowest
            index, `j`, less than `i`, where `E[j-1] - E[i-1] < xMax`. `iMax[i,1]` is returns the
            highest index, `j`, greater than `i`, where `E[i-1] - E[j-1] < xMax`. Indices 0 and L+1
            are reserved for the lower and upper energy range, `EB[0]` and `EB[1]`.
        """

        L  = E.shape[0]
        
        # Find iMax:
        iMax = self.findIMax(E, EB)
        
        # Level-spacing calculation:
        level_spacing_probs = np.zeros((L+2,L+2), dtype='f8')
        if self.G == 1:     # one spingroup --> no merge needed
            for i in range(L-1):
                X = E[i+1:iMax[i+1,0]-1] - E[i]
                level_spacing_probs[i+1,i+2:iMax[i+1,0]] = self.level_spacing_dists.f0(X)[:,0]
            # Boundary distribution:
            level_spacing_probs[0,1:-1]  = self.level_spacing_dists.f1(E - EB[0])[:,0]
            level_spacing_probs[1:-1,-1] = self.level_spacing_dists.f1(EB[1] - E)[:,0]
        else:               # multiple spingroups --> merge needed
            for i in range(L-1):
                X = E[i+1:iMax[i+1,0]-1] - E[i]
                prior_L = np.tile(prior[i,:], (iMax[i+1,0]-i-2, 1))
                prior_R = prior[i+1:iMax[i+1,0]-1,:]
                level_spacing_probs[i+1,i+2:iMax[i+1,0]] = self.levelSpacingMerge(X, prior_L, prior_R)
            # Boundary distribution:
            level_spacing_probs[0,1:-1]  = self.levelSpacingMergeBounds(E - EB[0], prior)
            level_spacing_probs[1:-1,-1] = self.levelSpacingMergeBounds(EB[1] - E, prior)

        # Error checking:
        if (level_spacing_probs == np.nan).any():   raise RuntimeError('Level-spacing probabilities have "NaN" values.')
        if (level_spacing_probs == np.inf).any():   raise RuntimeError('Level-spacing probabilities have "Inf" values.')
        if (level_spacing_probs <  0.0).any():      raise RuntimeError('Level-spacing probabilities have negative values.')
        
        # Verbose:
        if verbose: print('Finished level-spacing calculations.')

        # The normalization factor is duplicated in the prior. One must be removed: FIXME!!!!!
        level_spacing_probs /= self.lvl_dens_tot

        return level_spacing_probs, iMax
    
    def findIMax(self, E, EB:tuple):
        """
        Finds the approximation threshold indices, where the level-spacing probabilities are
        negligible according to `xMax` and `xMaxB`.

        Parameters:
        ----------
        E    :: float [L]
            All recorded resonance energies within the selected energy range, provided by `EB`.
        EB   :: float [2]
            The lower and upper boundaries on the recorded energy range.

        Returns:
        -------
        iMax :: int [L+2,2]
            An array of indices representing the outermost resonances where the level-spacing is
            not negligible as determined by `xMax` and `xMaxB`. `iMax[i,0]` is returns the lowest
            index, `j`, less than `i`, where `E[j-1] - E[i-1] < xMax`. `iMax[i,1]` is returns the
            highest index, `j`, greater than `i`, where `E[i-1] - E[j-1] < xMax`. Indices 0 and L+1
            are reserved for the lower and upper energy range, `EB[0]` and `EB[1]`.
        """

        L  = E.shape[0]
        
        iMax = np.full((L+2,2), -1, dtype='i4')

        # Lower boundary cases:
        for j in range(L):
            if E[j] - EB[0] >= self.xMaxB:
                iMax[0,0]    = j
                iMax[:j+1,1] = 0
                break

        # Intermediate cases:
        for i in range(L-1):
            for j in range(iMax[i,0]+1,L):
                if E[j] - E[i] >= self.xMax:
                    iMax[i+1,0] = j
                    iMax[iMax[i-1,0]:j+1,1] = i+1
                    break
            else:
                iMax[i:,0] = L+1
                iMax[iMax[i-1,0]:,1] = i+1
                break

        # Upper boundary cases:
        for j in range(L-1,-1,-1):
            if EB[1] - E[j] >= self.xMaxB:
                iMax[-1,1] = j
                iMax[j:,0] = L+1
                break

        return iMax

    def levelSpacingMerge(self, X, prior_L, prior_R):
        """
        A function that calculates the merged level-spacing probability density at a level-spacing
        of `X` with prior spingroup probabilities on the left and right resonances, `prior_L` and
        `prior_R`.

        This method was derived independently, but this distribution equaiton was first found by
        M. L. Mehta, but we have generalized it to include priors:
        http://home.ustc.edu.cn/~zegang/pic/Mehta-Random-Matrices.pdf (page 216-217; Eq. A22.10)

        Let `L` be the number of resonances and `G` be the number of (true) spingroups.

        Parameters:
        ----------
        X       :: float [L]
            Nearest-neighbor level-spacings.
        prior_L :: float [L,G]
            Prior spingroup probabilities for the left-hand-side resonances.
        prior_R :: float [L,G]
            Prior spingroup probabilities for the right-hand-side resonances.

        Returns:
        -------
        probs   :: float [L]
            probability densities from the merged level-spacing probability density function,
            evaluated at the given level-spacings, `X`.
        """

        f2, r1, r2 = self.level_spacing_dists.parts(X.reshape(-1,))
        c = np.prod(f2, axis=1)
        d = r2 * (r1 - r2)

        # Calculating normalization factor:
        Z = self.Z.reshape(1,self.G,self.G)
        wL = prior_L.reshape(-1,1,self.G)
        wR = prior_R.reshape(-1,self.G,1)
        norm = (wL @ Z @ wR)[:,0,0] # the normalization factor for the merged level-spacing distribution

        # Full probability calculation:
        wL = prior_L
        wR = prior_R
        probs = (c / norm) * ( \
                    np.sum(wL * r2, axis=1) \
                  * np.sum(wR * r2, axis=1) \
                  + np.sum(wL * wR * d, axis=1) \
                )
        return probs.reshape(X.shape)

    def levelSpacingMergeBounds(self, X, prior):
        """
        A function that calculates the merged level-spacing probability density at a distance of
        `X` from the boundary of the recorded enrgy range. The level-spacing distribution is
        informed by prior spingroup probabilities for the recorded resonance, `prior`.

        Let `L` be the number of resonances and `G` be the number of (true) spingroups.

        Parameters:
        ----------
        X     :: float [L]
            Distance between the resonance and the boundary of the recorded energy range.
        prior :: float [L,G]
            Prior spingroup probabilities for the recorded resonance.

        Returns:
        -------
        probs :: float [L]
            probability densities from the merged level-spacing probability density function,
            evaluated at the given level-spacings, `X`.
        """

        f2, r1, r2 = self.level_spacing_dists.parts(X.reshape(-1,))
        c = np.prod(f2, axis=1)

        # Calculating normalization factor:
        norm = prior @ self.ZB

        # Full probability calculation:
        probs = (c / norm) * np.sum(prior*r2, axis=1)
        return probs.reshape(X.shape)

# =================================================================================================
#     WigBayes Partition / Run Master:
# =================================================================================================

class RunMaster:
    f"""
    A wrapper over Encore responsible for partitioning spingroups, merging distributions, and
    combining spingroup probabilities. For 1 or 2 spingroups, partitioning and merging
    distributions is not necessary and RunMaster will pass to Encore. Once a RunMaster object has
    been initialized, the specific algorithm can be chosen (i.e. WigBayes, WigSample, etc.).

    ...
    """

    def __init__(self, E, EB:tuple,
                 level_spacing_dists:Distributions, false_dens:float=0.0,
                 Prior=None, log_likelihood_prior:float=None,
                 err:float=1e-9):
        """
        Initializes a RunMaster object.

        Parameters:
        ----------
        E                    :: float, array-like
            Resonance energies for the ladder.
        EB                   :: float [2]
            The ladder energy boundaries.
        level_spacing_dists  :: Distributions
            The level-spacing distributions object.
        false_dens                :: float
            The false level-density. Default = 0.0.
        Prior                :: float, array-like
            The prior probabilitiy distribution for each spingroup. Default = None.
        log_likelihood_prior :: float
            The log-likelihood provided from the prior. Default = None.
        err                  :: float
            A probability threshold where resonances are considered to be too far apart to be
            nearest neighbors.
        """
        
        # Error Checking:
        if type(level_spacing_dists) != Distributions:
            raise TypeError('The level-spacing distributions must be a "Distributions" object.')
        if not (0.0 < err < 1.0):
            raise ValueError('The probability threshold, "err", must be strictly between 0 and 1.')
        if len(EB) != 2:
            raise ValueError('"EB" must be a tuple with two elements: an lower and upper bound on the resonance ladder energies.')
        if EB[0] >= EB[1]:
            raise ValueError('EB[0] must be strictly less than EB[1].')
        
        self.E  = np.sort(E)
        self.EB = tuple(EB)
        self.level_spacing_dists = level_spacing_dists

        self.lvl_dens = np.concatenate((self.level_spacing_dists.lvl_dens, [false_dens]))

        self.L = len(E) # Number of resonances
        self.G = len(self.lvl_dens) - 1 # number of spingroups (not including false group)

        if Prior is None:
            self.Prior = np.tile(self.lvl_dens/self.lvl_dens_tot, (self.L,1))
        else:
            self.Prior = Prior
        self.log_likelihood_prior = log_likelihood_prior
        self.err = err
    
    @property
    def lvl_dens_tot(self):
        return np.sum(self.lvl_dens)
    @property
    def false_dens(self):
        return self.lvl_dens[-1]

    def mergePartitioner(s, partitions:list):
        """
        ...
        """

        n = len(partitions)

        # Merged level-spacing calculation:
        level_spacing_probs = np.zeros((s.L+2, s.L+2, n), 'f8')
        iMax = np.zeros((s.L+2, 2, n), 'i4')
        for g, group in enumerate(partitions):
            merger = Merger(s.level_spacing_dists[group], err=s.err)
            level_spacing_probs[:,:,g], iMax[:,:,g] = merger.findLevelSpacings(s.E, s.EB, s.Prior[:,group])

        # Merged prior calculation:
        prior_merged = np.zeros((s.L, n+1), 'f8')
        for g, group in enumerate(partitions):
            if hasattr(group, '__iter__'):
                prior_merged[:,g] = np.sum(s.Prior[:,group], axis=1)
            else:
                prior_merged[:,g] = s.Prior[:,group]
        prior_merged[:,-1] = s.Prior[:,-1]

        return level_spacing_probs, iMax, prior_merged

    def WigBayes(s, return_log_likelihood:bool=False, verbose:bool=False):
        """
        Returns spingroup probabilities for each resonance based on level-spacing distributions,
        and any provided prior.

        Parameters:
        ----------
        return_log_likelihood :: bool
            Determines whether to return the resonance ladder log-likelihood. Default = False.
        verbose :: bool
            The verbosity controller. Default = False.

        Returns:
        -------
        sg_probs :: int [L,G]
            The sampled IDs for each resonance and trial.
        """

        # 1 spingroup (merge not needed):
        if   s.G == 1:
            if verbose: print(f'Preparing level-spacings')
            level_spacing_probs_g, iMax_g, prior_g = s.mergePartitioner([[0]])
            if verbose: print(f'Finished level-spacing calculations')
            ENCORE = Encore(prior_g, level_spacing_probs_g, iMax_g)
            if verbose: print(f'Finished CP calculation')
            sg_probs = ENCORE.WigBayes()
            if verbose: print(f'Finished WigBayes calculation')

            if return_log_likelihood:
                log_likelihood = ENCORE.LogLikelihood(s.EB, s.false_dens, s.log_likelihood_prior)
                return sg_probs, log_likelihood
            else:
                return sg_probs
        
        # 2 spingroups (merge not needed):
        elif s.G == 2:
            if verbose: print(f'Preparing level-spacings')
            level_spacing_probs_g, iMax_g, prior_g = s.mergePartitioner([[0], [1]])
            if verbose: print(f'Finished level-spacing calculations')
            ENCORE = Encore(prior_g, level_spacing_probs_g, iMax_g)
            if verbose: print(f'Finished CP calculation')
            sg_probs = ENCORE.WigBayes()
            if verbose: print(f'Finished WigBayes calculation')

            if return_log_likelihood:
                log_likelihood = ENCORE.LogLikelihood(s.EB, s.false_dens, s.log_likelihood_prior)
                return sg_probs, log_likelihood
            else:
                return sg_probs

        # More than 2 spingroups (merge needed):
        else:
            sg_probs = np.zeros((s.L,3,s.G),dtype='f8')
            if return_log_likelihood:
                log_likelihood = np.zeros(s.G, dtype='f8')

            # Partitioning:
            for g in range(s.G):
                partition = [[g_ for g_ in range(s.G) if g_ != g], [g]]
                if verbose: print(f'Preparing for Merge group, {g}')
                level_spacing_probs_g, iMax_g, prior_g = s.mergePartitioner(partition)
                if verbose: print(f'Finished spingroup {g} level-spacing calculation')
                ENCORE = Encore(prior_g, level_spacing_probs_g, iMax_g)
                if verbose: print(f'Finished spingroup {g} CP calculation')
                sg_probs[:,:,g] = ENCORE.WigBayes()
                if verbose: print(f'Finished spingroup {g} WigBayes calculation')

                if return_log_likelihood:
                    # FIXME: I DON'T KNOW LOG Likelihood CORRECTION FACTOR FOR MERGED CASES! 
                    # lvl_dens_comb = np.array([s.lvl_dens[0,g], s.lvl_dens_tot-s.lvl_dens[0,g]]).reshape(1,-1)
                    log_likelihood[g] = ENCORE.LogLikelihood(s.EB, s.false_dens, s.log_likelihood_prior)

            # Combine probabilities for each merge case:
            combined_sg_probs = s.probCombinator(sg_probs)
            if return_log_likelihood:
                if verbose: print('Preparing for Merge group, 999!!!')
                level_spacing_probs_1, iMax_1, prior_1 = s.mergePartitioner([tuple(range(s.G))])
                if verbose: print('Finished spingroup 999 level-spacing calculation')
                ENCORE = Encore(prior_1, level_spacing_probs_1, iMax_1)
                if verbose: print('Finished spingroup 999 CP calculation')
                base_log_likelihood = ENCORE.LogLikelihood(s.EB, s.false_dens, s.log_likelihood_prior)
                combined_log_likelihood = s.logLikelihoodCombinator(log_likelihood, base_log_likelihood)
                if verbose: print('Finished!')
                return combined_sg_probs, combined_log_likelihood
            else:
                if verbose: print('Finished!')
                return combined_sg_probs
            
    def WigSample(s, trials:int=1, verbose:bool=False):
        """
        Returns random spingroup assignment samples based on its Bayesian probability.

        Parameters:
        ----------
        trials  :: int
            The number of trials of sampling the resonance ladder. Default = 1.
        verbose :: bool
            The verbosity controller. Default = False.

        Returns:
        -------
        samples :: int [L,trials]
            The sampled IDs for each resonance and trial.
        """

        # 1 spingroup (merge not needed):
        if s.G == 1:
            if verbose: print(f'Preparing level-spacings')
            level_spacing_probs_g, iMax_g, prior_g = s.mergePartitioner([[0]])
            if verbose: print(f'Finished level-spacing calculations')
            ENCORE = Encore(prior_g, level_spacing_probs_g, iMax_g)
            if verbose: print(f'Finished CP calculation')
            samples = ENCORE.WigSample(trials)
            if verbose: print(f'Finished WigBayes calculation')
            return samples
        
        # 2 spingroups (merge not needed):
        elif s.G == 2:
            if verbose: print(f'Preparing level-spacings')
            level_spacing_probs_g, iMax_g, prior_g = s.mergePartitioner([[0], [1]])
            if verbose: print(f'Finished level-spacing calculations')
            ENCORE = Encore(prior_g, level_spacing_probs_g, iMax_g)
            if verbose: print(f'Finished CP calculation')
            samples = ENCORE.WigSample(trials)
            if verbose: print(f'Finished WigBayes calculation')
            return samples

        # More than 2 spingroups (merge needed):
        else:
            raise NotImplementedError('WigSample for more than two spingroups has not been implemented yet.')

    def probCombinator(self, sg_probs):
        """
        Combines probabilities from various spingroup partitions.

        ...
        """

        combined_sg_probs = np.zeros((self.L,self.G+1), dtype='f8')
        for g in range(self.G):
            combined_sg_probs[:,g] = sg_probs[:,1,g]
        combined_sg_probs[:,-1] = np.prod(sg_probs[:,1,:], axis=1) * self.Prior[:,-1] ** (1-self.G)
        combined_sg_probs[self.Prior[:,-1]==0.0,  -1] = 0.0
        combined_sg_probs /= np.sum(combined_sg_probs, axis=1).reshape((-1,1))
        return combined_sg_probs

    def logLikelihoodCombinator(self, partition_log_likelihoods, base_log_likelihoods:float):
        """
        Combines log-likelihoods from from various partitions.
        """

        return np.sum(partition_log_likelihoods) - (self.G-1)*base_log_likelihoods