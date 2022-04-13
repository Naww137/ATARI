#!/bin/bash
#PBS -V
#PBS -q fill
#PBS -l nodes=1:ppn=1
#PBS -l pmem=3500mb

#### cd working directory (where you submitted your job)
cd ${PBS_O_WORKDIR}

#### load module
module load matlab

#### Executable Line
matlab -nodisplay -nodesktop -nojvm -nosplash -r 'run fit_main.m; exit'