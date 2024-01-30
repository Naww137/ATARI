Note's to Users
===============

Dependencies:

- Much of ATARI uses the resonance modelling & evaluation code SAMMY \cite{}. SAMMY is open source and can be coupled to ATARI simply by providing a path to the local SAMMY executable after build/install of SAMMY.

Standard units throughout the code:

=================  =========
  Quantity           Units
=================  =========
 Mass               amu
 Nuclear Radius     √barns
 Channel Radius     √barns
 Energy             eV
 Partial Widths     meV
=================  =========

Assumptions within the code:

- Penetrability for aggregate capture width is assumed to be 1 as per SAMMY, therefore partial capture widths are exactly double the reduced capture widths (:math:`\Gamma_\gamma=2\gamma_\gamma^2`).
- Reduced neutron widths, :math:`\gamma_n^2`, are related to the partial neutron widths, :math:`\Gamma_n`, by the equation :math:`\Gamma_n=2P(E_\lambda) \gamma_\gamma^2` as per SAMMY.

Under-developed behavior:
- Sammy ECSCM generation cannot be on a grid more dense that 498 points. A new reader function is needed.
