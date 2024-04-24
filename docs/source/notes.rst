Note's to Users
===============

This page summarizes some important qualities of ATARI. Reading this page is suggested to avoid misuse.

Standard Units
--------------

ATARI uses the following standard units throughout the code:

=================  =========
  Quantity           Units
=================  =========
 Mass               amu
 Nuclear Radius     √barns
 Channel Radius     √barns
 Energy             eV
 Partial Widths     meV
=================  =========

Code Assumptions, Notation, and Terminology
-------------------------------------------

- Penetrability for aggregate capture width is assumed to be 1 as per SAMMY, therefore partial capture widths are exactly double the reduced capture widths (:math:`\Gamma_\gamma=2\gamma_\gamma^2`).
- Reduced neutron widths, :math:`\gamma_n^2`, are related to the partial neutron widths, :math:`\Gamma_n`, by the equation :math:`\Gamma_n=2P(E_\lambda) \gamma_\gamma^2` as per SAMMY.

Under-developed Behavior
------------------------

- Sammy ECSCM generation cannot be on a grid more dense that 498 points. A new reader function is needed.
