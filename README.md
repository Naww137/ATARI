# ATARI
AI/ML Tool for Automated Resonance Identification

This is the repository for scripts, documents, and other resources relevant to the ATARI project, a part of Noah Walton's PhD research at the University of Tennessee. 
If you use this software, please cite the code using the citation link in the right hand pannel of this GitHub page and the appropriate journal publication detailing the methodology. 
The methodology for generating synthetic neturon time-of-flight data can be found `here <https://doi.org/10.1016/j.cpc.2023.108927>`.
The methodology for automated resonance fitting can be found `here <arXiv:2402.14122>`.
The initial demonstration for validating the automated fitting tool with the synthetic data approach can be found `here <https://doi.org/10.1016/j.anucene.2024.111081>`.


Installing Dependencies
=======================
ATARI has several dependencies. Most of the dependencies can be read from `requirements.txt`. To
install these dependencies, run the following line.

`pip install -r requirements.txt`

Most of ATARI's capabilities requires `SAMMY <https://code.ornl.gov/RNSD/SAMMY>`_ to be installed.
SAMMY is open source and can be coupled to ATARI simply by providing a path to the local SAMMY
executable after build/install of SAMMY. This is done for each script that is run.

Soon to come... a persistent link to SAMMY during install of the ATARI code...


Quick Installation
==================
ATARI can be installed quickly using pip:

`pip install --upgrade git+https://github.com/Naww137/ATARI.git`

Installing from a cloned directory
==================================
To clone ATARI, first migrate to the desired directly. Use the following command to clone the
ATARI directory:

`git clone https://github.com/Naww137/ATARI.git`

Migrate to the `ATARI` directory and run `pip install .`. Alternatively, if you plan to make
changes to the code, run `pip install -e .`. Note that the `-e` option may not work on
all systems. If your python environment is not active, run `python -m pip install .` or
`python -m pip install -e .`.

