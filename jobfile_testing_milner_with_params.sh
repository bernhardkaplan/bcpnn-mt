#!/bin/bash -l
# The -l above is required to get the full environment with modules

# The name of the script is myjob
#SBATCH -J sweep_params

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 0:29:00

# Number of cores to be allocated (multiple of 20)
#SBATCH -n 20

# Number of cores hosting OpenMP threads

#SBATCH -e error_file_testing.e
#SBATCH -o output_file_testing.o

# Run the executable named myexe 
# and write the output into my_output_file

echo "Starting at `date`"
export CRAY_ROOTFS=DSL

#. /opt/modules/default/etc/modules.sh
module swap PrgEnv-cray PrgEnv-gnu
module add nest/2.2.2
module add python

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cfs/milner/scratch/b/bkaplan/BCPNN-Module/build-module-100725
export PYTHONPATH=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages:/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages

aprun -n 20 -N 20 -S 10 python /cfs/milner/scratch/b/bkaplan/bcpnn-mt/main_test.py connection_matrix_20x16_taui5.dat connection_matrix_20x16_taui150.dat $1 > delme_test_$1 2>&1

echo "Stopping at `date`"

