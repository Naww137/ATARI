#!/bin/bash
#PBS -V
#PBS -q fill
#PBS -l nodes=1:ppn=8
#PBS -t 0-2%2

#### cd working directory (where you submitted your job)
cd ${PBS_O_WORKDIR}

#### load module
module load matlab

#### Executable Line
matlab -nodisplay -batch "baron_fit_rev1('/home/nwalton1/reg_perf_tests/perf_tests/staticladder/perf_test_staticladder.hdf5', ${PBS_ARRAYID})"
#### echo ${PBS_ARRAYID}