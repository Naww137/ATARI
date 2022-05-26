#!/bin/bash
#PBS -V
#PBS -q fill
#PBS -l nodes=1:ppn=8
#PBS -l pmem=3500mb
#PBS -t %%%array_specs%%%
#PBS -j oe

#### cd working directory (where you submitted your job)
cd ${PBS_O_WORKDIR}

#### load module
module load matlab

#### Executable Line
matlab -nodisplay -nodesktop -nosplash -r 'run %%%parm_name%%%_smpl_${PBS_ARRAYID}.m; exit'