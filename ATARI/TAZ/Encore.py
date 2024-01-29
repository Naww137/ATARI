#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
# import autograd.numpy as np

__doc__ = """
This file is responsible for the 1 and 2 spingroup classification algorithm. For more than 2
spingroups, the `Merger` class must be used to combine spingroups.
"""

class Encore:
    """
    A class for 1 or 2 spingroup classification. This class has many features:

    1. WigBayes: spingroup probabilities for each resonance, given the resonance ladder and mean
    parameters.

    2. WigSample: random sample of possible spingroup assignments. Likelihood of ladder sample
    correlates with the ladder spingroup assignment probability.
    
    3. ProbOfSample: probability of sampling the exact ladder of spingroup assignments.

    4. LogTotProb: log of the probability of sampling the resonance ladder regardless
    of spingroup, given the mean parameters. This is used for log-likelihood estimation for mean
    parameter estimation.

    Let "L" be the number of resonances in the ladder and "G" be the number of spingroups
    (excuding the false group).

    Initialization Variables:
    ------------------------
    prior               :: float [L+2, G+1]
        Prior probabilities on each resonance (with boundaries) and spingroup (with false group).

    level_spacing_probs :: float [L+2, L+2, G]
        All of the level spacings from position index 0 to index 1 with spingroup, index 2.

    iMax                :: int [L+2, 2, G]
        Maximum/minimum index for approximation threshold for each resonance (index 0) in each
        direction (index 1) for each spingroup (index 2).
    
    Internally Used Variables:
    -------------------------
    CP    :: float [L+2, L+2, 2]
        'Cumulative Probabilities' which are the building blocks for Bayesian probability
        calculations and probabilistic sampling.

    PW    :: float [L+2]
        'Placement Weights' which ensure CP does not grow or decay exponententially beyond the
        limits of floating-point numbers. PW is effectively a list of probability normalization
        factors for each resonance.

    TP    :: float
        'Total Probability' which is the probability density for the current arrangements of
        resonance energies (with PW removed for floating point limitation reasons).
    """

    # Error and debugging text:
    PW_INF_ERROR = """
Infinite placement weight ("PW") found. This implies a local probability of zero.
This should not happen unless the floating point limitations have been exceeded.

DEBUGGING VALUES:
Direction: {side}
Index:     {idx}
List A:    {AList}
List B:    {BList}
"""

    CP_BAD_ERROR = 'A cumulative probability ("CP") has a "{}" value.'

    TP_PERCENT_ERROR = """
In calculating the total probability ("TP"), there was a percent error of {p_error:.3f}%.
This exceeds the set limit of {p_thres:.3f}%. This error could be attributed to a bug or
numerical instability.
"""

    TP_error_threshold = 1.0 # % (percent)

    # ==================================================================================
    # Constructor
    # ==================================================================================
    def __init__(self, prior, level_spacing_probs, iMax):
        """
        Initialization ...

        ...
        """

        # Initialization:
        self.G = level_spacing_probs.shape[2]
        if self.G not in (1,2):
            raise NotImplementedError('ENCORE can only handle 1 or 2 spingroups and a false group at the moment.')
        self.L = np.int32(prior.shape[0])
        self.Prior = np.ones((self.L+2,self.G+1), dtype='f8')
        self.Prior[1:-1,:] = prior
        self.LSP = level_spacing_probs
        self.iMax = iMax

        # Calculating "CP" and "TP":
        if   self.G == 1:       self.__makeCP1()
        elif self.G == 2:       self.__makeCP2()
        self.__makeTP()

    # ==================================================================================
    # Make CP (1 spingroups and false group)
    # ==================================================================================
    def __makeCP1(s):
        """
        `__MakeCP1` calculates `CP` -- the 'Cumulative Probabilities' for the 1 spingroup case.
        `CP` is a three dimensional array that takes the form, `CP(i,dir)`, where `i` is the
        indices of the true spingroup and `dir` is the direction that the probabilities are
        accumulating. ...

        ...
        """

        L = s.L
        s.CP = np.zeros((L+2,2), dtype='f8')
        s.PW = np.zeros(L+2, dtype='f8')

        # Trivial CP & PW values:
        s.CP[ 0, 0] = 1.0
        s.CP[-1, 1] = 1.0
        s.PW[[0,-1]] = 1.0

        for iL in range(1,L+2):
            iR = L+1-iL

            # =================================================================================
            # Left-Hand Side:

            s.CP[iL,0] = s.CP[iL-1,0] * s.PW[iL-1] * s.Prior[iL-1,0] * s.LSP[iL-1,iL,0]
            mult_chain = 1.0
            for j in range(iL-2, s.iMax[iL,1,0]-1, -1):
                mult_chain *= s.Prior[j+1,1] * s.PW[j+1]
                if mult_chain == 0.0:   break   # Potential computation time improvement
                s.CP[iL,0] += s.CP[j,0] * mult_chain * s.PW[j] * s.Prior[j,0] * s.LSP[j,iL,0]
            
            # =================================================================================
            # Right-Hand Side:

            s.CP[iR,1] = s.CP[iR+1,1] * s.PW[iR+1] * s.Prior[iR+1,0] * s.LSP[iR,iR+1,0]
            mult_chain = 1.0
            for j in range(iR+2, s.iMax[iR,0,0]+1, 1):
                mult_chain *= s.Prior[j-1,1] * s.PW[j-1]
                if mult_chain == 0.0:   break   # Potential computation time improvement
                s.CP[iR,1] += s.CP[j,1] * mult_chain * s.PW[j] * s.Prior[j,0] * s.LSP[iR,j,0]
            
            # =====================================================================================
            # Finding PW:

            if 2*iL <= L+1:
                if s.CP[iL,0] == 0.0:
                    raise RuntimeError(s.PW_INF_ERROR.format(side='left', idx=iL, AList=s.CP[iL,0], BList='NaN'))
                s.PW[iL] = 1.0 / s.CP[iL,0]

            if 2*iL <= L:
                if s.CP[iR,1] == 0.0:
                    raise RuntimeError(s.PW_INF_ERROR.format(side='right', idx=iR, AList=s.CP[iR,1], BList='NaN'))
                s.PW[iR] = 1.0 / s.CP[iR,1]

    # ==================================================================================
    # Make CP (2 spingroups and false group)
    # ==================================================================================
    def __makeCP2(s):
        """
        `__MakeCP2` calculates `CP` -- the 'Cumulative Probabilities'. `CP` is a three dimensional
        array that takes the form, `CP(iA,iB,dir)`, where `iA` and `iB` are the indices of 'A'
        and 'B' and `dir` is the direction. `CP(iA,iB,dir)` is the probability of the spingroup
        assignment distribution on the set of resonances with indices less that or equal to
        `max(iA,iB)` for `dir = 1` or greater than or equal to `min(iA,iB)` for `dir = 2` that
        meet the following conditions:
            (1) `iA` is the index of the right-most (`dir = 1`)
                or left-most (`dir = 2`) resonance with spingroup A.
            (2) `iB` is the index of the right-most (`dir = 1`)
                or left-most (`dir = 2`) resonance with spingroup B.

        `CP` is calculated through an iterative process, hince the name. The true value of `CP`
        has the tendency to exponentially grow or decay through each iteration, therefore, we use
        'Placement Weights' (`PW`) to renormalize the value to an acceptable floating-point range.
        """

        L = s.L
        s.CP = np.zeros((L+2,L+2,2), dtype='f8')
        s.PW = np.zeros(L+2, dtype='f8')

        # Trivial CP values:
        s.CP[ 0, 0, 0] = 1.0 ;      s.CP[-1, -1, 1] = 1.0
        # s.CP[ 1, 0, 0], s.CP[ 0, 1, 0] = s.LSP[ 0, 1, :]
        # s.CP[-2,-1, 1], s.CP[-1,-2, 1] = s.LSP[-2,-1, :]

        s.PW[[0,-1]] = 1.0
        # denom = s.CP[ 1, 0, 0] + s.CP[ 0, 1, 0]
        # s.PW[1]  = 1.0 / denom
        # denom = s.CP[-2,-1, 1] + s.CP[-1,-2, 1]
        # s.PW[-2] = 1.0 / denom


        jMax  = np.array([[0, L+1], [0, L+1]], dtype='u4')
        for i1L in range(1,L+2):
            i1R = L+1-i1L

            jMax[:,0] = s.iMax[i1L,1,:]
            jMax[:,1] = s.iMax[i1R,0,:]
            i2Max = jMax[1::-1,:]
            for t in range(2):
                # =================================================================================
                # Left-Hand Side Iteration:

                idx = range(i1L-1, jMax[t,0]-1, -1)
                ls = s.LSP[idx, i1L, t]
                for i2L in range(i1L-1, i2Max[t,0]-1, -1):

                    # Finding J1 and J2:
                    if (jMax[t,0] > i2L) or (i2L == 0):
                        j1 = idx ;      ls1 = ls
                        j2 = []  ;      ls2 = []
                    else:
                        j1 = range(i1L-1, i2L, -1)         ;    ls1 = ls[:i1L-i2L-1]
                        j2 = range(i2L-1, jMax[t,0]-1, -1) ;    ls2 = ls[i1L-i2L:]
                        
                    # New CP Iteration:
                    s.__CPIter(j1,j2,ls1,ls2,i1L,i2L,t,1)

                # =================================================================================
                # Right-Hand Side Iteration:

                idx = range(i1R+1, jMax[t,1]+1)
                ls  = s.LSP[i1R, idx, t]
                for i2R in range(i1R+1, i2Max[t,1]+1):

                    # Finding J1 and J2:
                    if (jMax[t,1] < i2R) or (i2R == L+1):
                        j1 = idx ;      ls1 = ls
                        j2 = []  ;      ls2 = []
                    else:
                        j1 = range(i1R+1, i2R)         ;    ls1 = ls[:i2R-i1R-1]
                        j2 = range(i2R+1, jMax[t,1]+1) ;    ls2 = ls[i2R-i1R:]

                    # New CP Iteration:
                    s.__CPIter(j1,j2,ls1,ls2,i1R,i2R,t,-1)

            # =====================================================================================
            # Finding PW:

            # Left-Hand Side PW Calculation:
            if 2*i1L <= L+1:
                idxB = range(i2Max[0,0], i1L)
                idxA = range(i2Max[1,0], i1L)
                denom = np.sum(s.CP[i1L,idxB,0]) + np.sum(s.CP[idxA,i1L,0])
                if denom == 0.0:
                    raise RuntimeError(s.PW_INF_ERROR.format(side='left', idx=i1L, AList=s.CP[idxA,i1L,0], BList=s.CP[i1L,idxB,0]))
                s.PW[i1L] = 1.0 / denom

            # Right-Hand Side PW Calculation:
            if 2*i1L <= L:
                idxB = range(i1R+1, i2Max[0,1]+1)
                idxA = range(i1R+1, i2Max[1,1]+1)
                denom = np.sum(s.CP[i1R,idxB,1]) + np.sum(s.CP[idxA,i1R,1])
                if denom == 0.0:
                    raise RuntimeError(s.PW_INF_ERROR.format(side='right', idx=i1R, AList=s.CP[idxA,i1R,1], BList=s.CP[i1R,idxB,1]))
                s.PW[i1R] = 1.0 / denom

    # ==================================================================================
    # CP Iteration Function
    # ==================================================================================
    def __CPIter(s, J1, J2, ls1, ls2, i1, i2, t, sgn):
        """
        `__CPIter` is a function that calculates the next `CP` element using an iterative process.
        
        ...
        """

        dir = int(sgn == -1)
        cp_out = 0.0

        # Preceeding i2:
        if J1:
            if t:   prod1 = ls1 * s.CP[i2,J1,dir]
            else:   prod1 = ls1 * s.CP[J1,i2,dir]
            
            j = J1[0]
            mult_chain = s.PW[j]
            cp_out = mult_chain * prod1[0] * s.Prior[j,t]
            for j, p1 in zip(J1[1:], prod1[1:]):
                mult_chain *= s.PW[j] * s.Prior[j+sgn,2]
                if mult_chain == 0.0:   break   # Potential computation time improvement
                cp_out += mult_chain * p1 * s.Prior[j,t]

        # Behind i2:
        if J2:
            if i2+sgn == i1:    mult_chain  = s.PW[i2] * s.Prior[i2,1-t]
            elif i1 == i2:      mult_chain  = 1.0
            else:               mult_chain *= s.PW[i2] * s.Prior[i2,1-t] * s.Prior[i2+sgn,2]

            if t:   cp_out += mult_chain * np.sum(ls2 * s.CP[i2,J2,dir])
            else:   cp_out += mult_chain * np.sum(ls2 * s.CP[J2,i2,dir])

        # Error Checks:
        if   cp_out == math.inf:
            raise RuntimeWarning(s.CP_BAD_ERROR.format('Inf'))
        elif cp_out == math.nan:
            raise RuntimeWarning(s.CP_BAD_ERROR.format('NaN'))
        
        # Assigning CP:
        if t:   s.CP[i2,i1,dir] = cp_out
        else:   s.CP[i1,i2,dir] = cp_out

    # ==================================================================================
    # Find Total Probability Factor
    # ==================================================================================
    def __makeTP(s):
        """
        `__makeTP` is a function that calculates the total probability factor, `TP` using `CP`
        and `PW` found in the function, `__makeCP1` or `__makeCP2`. `TP` can be found by
        calculating `CP[ -1, -1, 0]` or `CP[ 0, 0, 1]`. Each can be performed by summing `CP`
        across the first or second spingroup. Therefore, there are a four ways to calculate `TP`.
        We accept the mean of the four calculation as the total probability and find the percent
        error for error-checking purposes.
        """

        # =====================================================================================
        # Finding CP[ -1, -1, 0] = CP[ 0, 0, 1] = TP:

        # Alternative solution:
        # for t in range(2):
        #     J2  = range(s.L, s.iMax[s.L+1,1,t]-1, -1)
        #     ls2 = s.LSP[J2, s.L+1, t]
        #     s.__CPIter([], J2, [], ls2, s.L+1, s.L+1, t, 1)

        #     J2  = range(1, s.iMax[0,0,t]-1)
        #     ls2 = s.LSP[0, J2, t]
        #     s.__CPIter([], J2, [], ls2, 0, 0, t, -1)

        if s.G == 2:
            TPs = np.empty((2,2), dtype='f8')
            TPs[0,0] = np.sum(s.LSP[:-1, -1, 0] * s.CP[:-1, -1, 0])
            TPs[1,0] = np.sum(s.LSP[:-1, -1, 1] * s.CP[-1, :-1, 0])
            TPs[0,1] = np.sum(s.LSP[  0, 1:, 0] * s.CP[ 1:,  0, 1])
            TPs[1,1] = np.sum(s.LSP[  0, 1:, 1] * s.CP[ 0,  1:, 1])
            s.CP[-1,-1,0], s.CP[0,0,1] = np.mean(TPs, axis=0)

        # =====================================================================================

        if   s.G == 1:      TP1 = s.CP[0,1];    TP2 = s.CP[-1,0]
        elif s.G == 2:      TP1 = s.CP[0,0,1];  TP2 = s.CP[-1,-1,0]
        
        # Taking the average of the two sides for the total probability factor:
        s.TP = (TP1 + TP2) / 2

        # If the two sides are past a threshold percent difference, raise a warning:
        percent_error = 100.0 * abs(TP1 - TP2) / s.TP
        if percent_error > s.TP_error_threshold:
            print(RuntimeWarning(s.TP_PERCENT_ERROR.format(p_error=percent_error, p_thres=s.TP_error_threshold)))

    # ==================================================================================
    # Level Spacing Bayes Theorem
    # ==================================================================================
    def WigBayes(s):
        """
        `WigBayes` finds the probability on each spingroup assignment of each resonance, informed
        by the level-spacing probability distributions. This is performed using the calculated `CP`
        and `TP` probabilities and Bayes' Theorem.

        ...
        """

        # Error checking:
        if s.G not in (1, 2):
            raise NotImplementedError('Currently "WigBayes" can only handle 1 or 2 spingroups and a False group. \
                                        Merge spingroups to get probabilities on more spingroups.')

        L = s.L
        sp = np.zeros((L,s.G), dtype='f8') # No longer need the edge cases, so the bounds are 0:L instead of 0:L+2

        if s.G == 1:
            sp[:,0] = s.CP[1:-1,0] * s.CP[1:-1,1]

        elif s.G == 2:
            for t in range(2):
                for iL in range(L):
                    iRMax = s.iMax[iL,0,1-t]
                    lsp = s.LSP[iL, iL+2:iRMax+1, 1-t]
                    for i in range(iL+1, iRMax):
                        idxR = range(i+1, iRMax+1)
                        if t:   sp[i-1,1] += s.CP[iL,i,0] * np.sum(lsp[i-iL-1:] * s.CP[idxR,i,1])
                        else:   sp[i-1,0] += s.CP[i,iL,0] * np.sum(lsp[i-iL-1:] * s.CP[i,idxR,1])
                    # if t:   sp[iL:iRMax-1,1] += np.array([s.CP[iL,i,0] * np.sum(ls[i-iL-1:] * s.CP[i+1:iRMax+1,i,1]) for i in range(iL+1, iRMax)])
                    # else:   sp[iL:iRMax-1,0] += np.array([s.CP[i,iL,0] * np.sum(ls[i-iL-1:] * s.CP[i,i+1:iRMax+1,1]) for i in range(iL+1, iRMax)])

        # Factors shared throughout the sum including the normalization factor, "TP":
        sp *= s.Prior[1:-1,:s.G] * s.PW[1:-1].reshape(-1,1) / s.TP

        # False probabilities appended by taking one minus the other probabilities:
        posterior = np.append(sp, (1.0 - np.sum(sp, axis=1)).reshape(-1,1), 1)
        return posterior

    # ==================================================================================
    # Sample Spacing-Updated Spin-Groups
    # ==================================================================================
    def WigSample(s, trials:int=1):
        """
        `WigSample` randomly samples spingroup assignments based its Bayesian probability. This is
        performed by sampling spingroup assignments for each resonance one at a time from low-to-
        high energy, indexed by `i`. Using the lower energy resonaces with determined spingroup
        assignments and the `CP` probabilities, representing the possibilities for the the higher
        energy resonances, we can get a probability on resonance `i` for each spingroup.
        
        ...
        """

        # Error checking:
        if s.G != 2:
            raise NotImplementedError("Currently sampling only works with 2 spingroups.")

        L = s.L
        ssg  = np.zeros((L,trials), dtype='u1') # The sampled spingroups

        Last = np.zeros((2,trials), dtype='u4') # Latest indices for each spingroup, for each trial
        rand_nums = np.random.rand(L, trials)   # Random numbers used to sample the spingroups

        iMax = int(3)
        sp = np.zeros((trials,3), dtype='f8')
        for i in range(L):
            i1 = i + 1

            # FIXME: THE APROXIMATION THRESHOLD NEEDS UPDATING!!!
            while any(s.LSP[i1,iMax,:]) and (iMax <= L):
                iMax += 1
            Idx = range(i1+1,iMax+1)
            l = iMax - i1

            ls1 = np.zeros((1,trials,2),dtype='f8')
            ls2 = np.zeros((l,trials,2),dtype='f8')
            for t in range(2):
                ls1[0,:,t] = s.LSP[Last[t,:],i1,t].T
                for tr in range(trials):
                    ls2[:,tr,t] = s.LSP[Last[1-t,tr],Idx,1-t]
            cp = np.concatenate((s.CP[i1,Idx,1].reshape(-1,1,1), s.CP[Idx,i1,1].reshape(-1,1,1)), axis=2)
            sp[:,:2] = (s.Prior[i1,:2].reshape(1,1,2)*ls1*np.sum(ls2*cp, axis=0))[0,:,:]
            
            sp[:,2]  = np.array([s.Prior[i1,2] * np.sum(ls2[:,tr,0].reshape(-1,1) \
                                                      * ls2[:,tr,1].reshape(1,-1) \
                                                      * s.CP[Idx,Idx,1], axis=(0,1)) \
                                                        for tr in range(trials)])

            sp_sum = np.sum(sp, axis=1)
            p2       = sp[:,2] / sp_sum      # false probability
            p1       = sp[:,1] / sp_sum + p2 # false or second spingroup probability

            # Sampling New Spin-groups:
            ssg[i,:] =  np.int_(rand_nums[i,:] <= p2) + np.int_(rand_nums[i,:] <= p1)

            # Rewriting the last used resonance of spingroup sg:
            for tr,sg in enumerate(ssg[i,:]):
                if sg != 2:     Last[sg,tr] = i1

        return ssg
    
    # ==================================================================================
    # Sample Probability
    # ==================================================================================
    def ProbOfSample(s, spingroup_assignments) -> float:
        """
        `ProbOfSample` finds the probability of sampling the given spingroup assignments.

        ...
        """

        # NOTE: This needs checking!!!
        likelihood = 1.0
        if   s.G == 1:
            raise NotImplementedError('Single spingroup sampling probability has not been implemented yet.')
        
        elif s.G == 2:
            last_res = np.zeros(2,'i4')
            for i, sg in enumerate(spingroup_assignments):
                j = last_res[sg]
                likelihood *= s.Prior[i+1,sg] * s.PW[i+1] * s.LSP[j,i+1,sg]
                last_res[sg] = i+1
            likelihood *= s.LSP[last_res[0],-1,0] * s.LSP[last_res[1],-1,1]
        
        prob_of_sampling = likelihood / s.TP
        return prob_of_sampling

    # ==================================================================================
    # Log Likelihood
    # ==================================================================================
    def LogLikelihood(self, EB:tuple, lvl_dens_false:float, log_likelihood_prior:float=None) -> float:
        """
        `LogLikelihood` gives a criterion for the likelihood of sampling the given energies and
        widths, assuming correct information was given (i.e. mean level spacing, mean neutron
        width, etc.). This can be used with gradient descent to optimize mean parameters.
        """

        dE = EB[1]-EB[0]
        log_likelihood = math.log(self.TP) - np.sum(np.log(self.PW)) - lvl_dens_false * dE

        # Prior log likelihood:
        if log_likelihood_prior is not None:
            log_likelihood += log_likelihood_prior

        return log_likelihood

    @staticmethod
    def LikelihoodString(log_likelihood:float, sigfigs:int=3) -> str:
        """
        Prints the likelihood given the log likelihood. The likelihood sometimes cannot be
        given directly since the exponential term could exceed floating-point limitations;
        for this reason, likelihood is given by a custom scientific notation.
        """

        # Turn to into log10:
        log_likelihood /= math.log(10)

        out_str = f'{{0:.{sigfigs}f}}e{{1}}'
        exponent    = math.floor(log_likelihood)
        significand = 10 ** (log_likelihood % 1.0)
        return out_str.format(significand, exponent)
    
    @staticmethod
    def ExpectedLogLikelihood(EB:tuple, lvl_densities:np.ndarray, log_likelihood_prior_exp:float=None) -> float:
        """
        ...
        """

        raise NotImplementedError('Expected log likelihood estimation has not been implemented yet.')

        dE = EB[1] - EB[0]

        # Prior log likelihood:
        if log_likelihood_prior_exp is not None:
            log_likelihood_exp += log_likelihood_prior_exp

        return log_likelihood_exp
    
# ==================================================================================
# Maximum-Likelihood Assignments
# ==================================================================================

def wigMaxLikelihood2(prior, level_spacing_probs, iMax, threshold:float=math.inf):
    """
    ...
    """
    spingroups   = []
    likelihoods  = []
    last_indices = []
    raise NotImplementedError()

# ==================================================================================
# Brute Force Algorithms
# ==================================================================================

def wigBayesBruteForce(E, distributions, false_dens:float, prior=None):
    """
    ...
    """
    raise NotImplementedError()