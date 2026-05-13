# List of Unit Tests for ATARI

# Test Data Management and Utilities

### Test ATARIO
- ...

### Test PiTFAll
- ...

### Test Utility Statistics
- Test that corr2cov works for a known case.
- Test that cov2corr works for a known case.

# Test Syndat

### Test Syndat
- ...

### Test Measurement Covariance
- ...

# Theory, Distributions, and Resonance Statistics

### Test RMatrix Theory
- Test that the explicit and implicit equations for penetrability match. (NOT IMPLEMENTED YET)

### Test Distributions
- Test that each Distribution PDF integrates to 1.
- Test that each Distribution has a mean matching the provided mean.
- Test that each Distribution's CDF and SF are inversions.
- Test that the partial integral of each Distribution PDF is the CDF.

### Test Level Spacing Distributions
- Test that each SpacingDistribution f0, f1, and f2 integrates to 1.
- Test that each SpacingDistribution f0 has a mean matching the provided mean.
- Test that the partial integral of each SpacingDistribution's f0 and f1 is f1 and f2 when scaled appropriately.
- Test that iF0 and iF1 are the inverse CDFs of f0 and f1.
- Test that Poisson distributions merge to another Poisson distribution with known level-density.

### Test Resonance Generator
- Test that the resonance sampler has level-spacings following Wigner distribution.
- Test that the resonance sampler has neutron widths following Porter-Thomas distribution.
- Test that the resonance sampler has capture widths following Porter-Thomas distribution.
- Test that the GOE, GUE, and GSE resonance sampler has resonances uniformly spaced in energy.
- Test that the GOE, GUE, and GSE resonance sampler has levels with an expected Dyson-Mehta Delta-3 statistic.
- Test that the resonance sampler with missing resonances has level-spacings following the missing resonance level-spacing distribution. (NOT INCORPORATED YET)
- Test that the Brody distribution resonance sampler has level-spacings following Brody distribution. (NOT INCORPORATED YET)
- Test that merged levels from the resonance sample have level-spacings following the merged level-spacing distribution. (NOT INCORPORATED YET)

### Test PTBayes
- Test that the correct assignment rate matches the assignment probabilities within statistical error for both gamma on and off. (NOT IMPLEMENTED YET)

# TAZ

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

### Test Empirical False Width Distribution
- ???