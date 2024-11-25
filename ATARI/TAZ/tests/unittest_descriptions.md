# List of Unit Tests for TAZ

### Test Theory
- Compare TAZ k_wavenumber to ATARI k_wavenumber (NOT IMPLEMENTED YET)
- Compare TAZ penetrabilities to ATARI penetrabilities
- Test that the explicit and implicit equations for penetrability match. (NOT IMPLEMENTED YET)

### Test Distributions
- Test that each Distribution integrate correctly, have correct mean, etc.

### Test Spacing Distributions
- Test that each SpacingDistribution integrate correctly, have correct mean, etc.
- Test that Poisson distributions merge to another Poisson distribution with known level-density.

### Test Samplers
- Test that the sampling algorithms follow expected distributions.
- Test merged distribution as well.

### Test PTBayes
- Test that the correct assignment rate matches the assignment probabilities within statistical error for both gamma on and off. (NOT IMPLEMENTED YET)

### Test WigBayes
- Test that False probabilities are zero when false level-density is zero.
- Test that a 2 spingroup case with a small second level-density converges to the 1 spingroup case.
- Test that a 3 spingroup case with a small third level-density converges to the 2 spingroup case. (NOT IMPLEMENTED YET)
- Test that providing Poisson distributions to WigBayes will return the same as the prior.
- Test that the correct assignment rate matches the assignment probabilities within statistical error.
- Test that spingroups are correctly merged. (NOT IMPLEMENTED YET)

### Test WigSample
- Test that WigSample returns spingroups with the correct frequency based on the underlying level-densities. (this is a poor unit test)
- Test that WigSample returns spingroups that produce the underlying distribution. Verify with Chi-square test. (this is a poor unit test)

### Test WigMaxLikelihoods
- Test that WigMaxLikelihoods returns the B-most likely prior ladders when provided Poisson distributions.
- Test case with identical mean parameters and different distributions. There should be spingroup symmetry.

### Test ProbOfSample
- ???

### Test Mean Parameter Estimation
- Test that the value and uncertainty for each mean parameter is statistically valid. (NOT IMPLEMENTED YET)