#!/bin/bash
#PBS -V
#PBS -q fill
#PBS -l nodes=1:ppn=1

#### cd working directory (where you submitted your job)
cd ${PBS_O_WORKDIR}

#### Executable Line
python3 csv_to_hdf5.py