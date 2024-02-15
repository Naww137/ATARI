------------------------------------------
:mod:`AutoFit` -- Automated Fitting Module
------------------------------------------
This module can be used to automatically fit experimental resonance data with no dependence on a prior.
The goal is for this to reduce the manual workload and potential sources of bias in a resonance evaluators workflow, thus improving reproducibility and the time it takes to produce a resonance evaluation.
This methodology, outlined in this journal publication (**ref**), leverages the resonance modelling code SAMMY (**ref**).
Consequently, this module in tightly coupled to the sammy_interface module of ATARI.


Control Class
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: class.rst

    ATARI.AutoFit.control.AutoFit_Control
    ATARI.AutoFit.control.AutoFitOPT
    ATARI.AutoFit.control.AutoFitOUT


Initial Feature Bank
--------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: class.rst

    ATARI.AutoFit.initial_FB_solve.InitialFBOPT
    ATARI.AutoFit.initial_FB_solve.InitialFB
    ATARI.AutoFit.initial_FB_solve.InitialFBOUT
    

