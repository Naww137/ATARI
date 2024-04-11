Using SAMMY
===========

`SAMMY <https://code.ornl.gov/RNSD/SAMMY>`_ is used to calculate R-matrix quantities for ATARI.
Additionally, ATARI uses SAMMY's built-in resolution functions. An example of SAMMY being used
can be found in :code:`examples/SAMMY_Interface.ipynb`. This Jupyter notebook can also be used
to verify that SAMMY has been installed successfully.

When using SAMMY, it is important to provide the path to the SAMMY executable file, `sammyexe`,
as an argument in the `SammyRunTimeOptions` class. The path is structured as shown below.

:code:`<path-to-sammy>/SAMMY/sammy/build/bin/sammy`