# The name of the script is neuron_job
#PBS -N spiking-bcpnn

#PBS -l walltime=1:30:00
#set which time allocation should be charged for this job

#(only needed if you belong to more than one time allocation)
#PBS -A 025-12-6

# Number of cores to be allocated is 24
#PBS -l mppwidth=96

#PBS -e error_file_python.e
#PBS -o output_file_python.o

# Change to the work directory
cd $PBS_O_WORKDIR

echo "Starting at `date`"
export CRAY_ROOTFS=DSL

. /opt/modules/default/etc/modules.sh
module swap PrgEnv-pgi PrgEnv-gnu
module add nest
module add site-python

export PYTHONPATH=/pdc/vol/nest/2.2.2/lib64/python2.6/site-packages:/pdc/vol/python/packages/site-python-2.6/lib64/python2.6/site-packages:/cfs/klemming/nobackup/b/bkaplan/PythonPackages/lib64/python2.6/site-packages/

aprun -n 96 python /cfs/nobackup/b/bkaplan/bcpnn-mt/main_training.py > delme_bcpnn_output_0 2>&1

echo "Stopping at `date`"

