#!/bin/bash
#PBS -V
#PBS -q fill
#PBS -l nodes=1:ppn=8
#PBS -t 100-499%25

#### cd working directory (where you submitted your job)
cd ${PBS_O_WORKDIR}

#### load module
module load matlab

#### Executable Line
matlab -nodisplay -batch "baron_fit_rev1('./perf_test_baron.hdf5', ${PBS_ARRAYID})"
#### echo ${PBS_ARRAYID}