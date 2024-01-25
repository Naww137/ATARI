Note's to Users
===============

Dependencies:
- Much of ATARI uses the resonance modelling & evaluation code SAMMY \cite{}. SAMMY is open source and can be coupled to ATARI simply by providing a path to the local SAMMY executable after build/install of SAMMY.
- 

Units throughout the code:
- Widths are all in meV while energy is in eV
- Scattering radii are in femtometers

Assumptions within the code:
- Penetrability for aggregate capture width is assumed to be 1, therefore partial capture widths are exactly double the reduced capture widths.
- 

Under-developed behavior:
- Sammy ECSCM generation cannot be on a grid more dense that 498 points. A new reader function is needed.
