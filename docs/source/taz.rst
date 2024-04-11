TAZ Guide
=========
**TAZ** is an algorithm for determining spingroup assignment using resonance parameter
distributions such as Wigner's level-spacing distribution and the Porter-Thomas resonance width
distribution. The **TAZ** algorithm is composed of a two-part Bayesian framework. First, the
`PTBayes` algorithm finds spingroup assignment probabilities based on resonance widths. Next, the
`Encore` framework takes a prior (usually the `PTBayes` probabilities) and applies Wigner
distribution, creating a level-spacing informed posterior.

`Encore` provides three useful results depending on the functions used. First, `WigBayes` returns
the spingroup probabilities for each individual resonance. `WigSample` samples resonance spingroup
assignments based on the assignment ladder's relative likelihood. Lastly, `LogLikelihood` returns
the log-likelihood of the ladder being sampled with the provided parameter distributions,
*regardless of spingroup assignment*.

Detailed Description
--------------------
First, the level-spacing distribution is defined as a `Distributions` object. This object stores
all relevant information regarding level-spacing distributions.

`RunMaster` must be used for preprocessing and post-processing for `Encore`. `RunMaster` creates a
`Merger` object. The `Merger` object is used when more than 2 spingroups are needed. In this case,
`Merger` combines `Distribution` objects of various spingroups as if they are the same spingroup,
creating new level-spacing distributions. `Merger` then provides the calculated level-spacings
which are then processed by `Encore`.