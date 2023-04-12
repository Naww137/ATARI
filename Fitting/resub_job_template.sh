#!/bin/bash
#PBS -V
#PBS -q fill
#PBS -l nodes=1:ppn=8


#### cd working directory (where you submitted your job)
cd ${PBS_O_WORKDIR}

#### load module
module load matlab

#### Executable Line
matlab -nodisplay -batch 
#### echo ${PBS_ARRAYID}