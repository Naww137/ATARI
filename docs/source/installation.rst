Quick Installation
==================
ATARI can be installed quickly using pip:

:code:`pip install --upgrade git+https://github.com/Naww137/ATARI.git`

Clone ATARI
===========
To clone ATARI, first migrate to the desired directly. Use the following command to clone the
ATARI directory:

:code:`git clone https://github.com/Naww137/ATARI.git`

Migrate to the `ATARI` directory and run :code:`pip install .`. Alternatively, if you plan to make
changes to the code, run :code:`pip install -e .`. Note that the :code:`-e` option may not work on
all systems. If your python environment is not active, run :code:`python -m pip install .` or
:code:`python -m pip install -e .`.

Installing Dependencies
=======================
ATARI has several dependencies. Most of the dependencies can be read from `requirements.txt`. To
install these dependencies, run the following line.

:code:`pip install -r requirements.txt`

Most of ATARI's capabilities requires `SAMMY <https://code.ornl.gov/RNSD/SAMMY>`_ to be installed.
SAMMY is open source and can be coupled to ATARI simply by providing a path to the local SAMMY
executable after build/install of SAMMY.

...