#!/bin/bash -l
# The -l above is required to get the full environment with modules

# The name of the script is myjob
#SBATCH -J OCT_training

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 2:30:00

# Number of cores per node to be allocated
#SBATCH -n 120

#SBATCH -e error_file.e
#SBATCH -o output_file.o

# Run the executable named myexe 
# and write the output into my_output_file

echo "Starting at `date`"
export CRAY_ROOTFS=DSL

#. /opt/modules/default/etc/modules.sh
module swap PrgEnv-cray PrgEnv-gnu
module add nest
module add python

PYNNPATH=/cfs/milner/scratch/b/bkaplan/PythonPackages/lib64/python2.6/site-packages/
# for the BCPNN module
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cfs/milner/scratch/b/bkaplan/BCPNN-Module/build-module-100725
export PYTHONPATH=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages:/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages:$PYNNPATH

aprun -n 120 python /cfs/milner/scratch/b/bkaplan/bcpnn-mt/NetworkSimModule.py > delme_training 2>&1

echo "Stopping at `date`"
