#!/bin/bash -l
# The -l above is required to get the full environment with modules
# The name of the script is myjob
#SBATCH -J plotting

# Only 1 hour wall-clock time will be given to this job
#SBATCH -t 00:29:00

# Number of cores per node to be allocated
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=20

#SBATCH -e error_file_plotting.e
#SBATCH -o output_file_plotting.o
#SBATCH --mail-type=END,FAIL

# Run the executable named myexe 
# and write the output into my_output_file

echo "Starting at `date`"
export CRAY_ROOTFS=DSL

#. /opt/modules/default/etc/modules.sh
module swap PrgEnv-cray PrgEnv-gnu
module add python
module add nest/2.2.2

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cfs/milner/scratch/b/bkaplan/BCPNN-Module/build-module-100725
export PYTHONPATH=/pdc/vol/nest/2.2.2/lib/python2.7/site-packages:/pdc/vol/python/2.7.6-gnu/lib/python2.7/site-packages

aprun -n 1 -N 1 python WeightAnalyser.py TrainingSim_Cluster_asymmetricTauij__50x2x1_0-400_taui5_nHC20_nMC4_vtrain1.00-1.0 > delme_plotting 2>&1 
aprun -n 1 -N 1 python WeightAnalyser.py TrainingSim_Cluster_asymmetricTauij__50x2x1_0-400_taui10_nHC20_nMC4_vtrain1.00-1.0 > delme_plotting 2>&1 
aprun -n 1 -N 1 python WeightAnalyser.py TrainingSim_Cluster_asymmetricTauij__50x2x1_0-400_taui20_nHC20_nMC4_vtrain1.00-1.0 > delme_plotting 2>&1 
aprun -n 1 -N 1 python WeightAnalyser.py TrainingSim_Cluster_asymmetricTauij__50x2x1_0-400_taui50_nHC20_nMC4_vtrain1.00-1.0 > delme_plotting 2>&1 
aprun -n 1 -N 1 python WeightAnalyser.py TrainingSim_Cluster_asymmetricTauij__50x2x1_0-400_taui100_nHC20_nMC4_vtrain1.00-1.0 > delme_plotting 2>&1 
aprun -n 1 -N 1 python WeightAnalyser.py TrainingSim_Cluster_asymmetricTauij__50x2x1_0-400_taui150_nHC20_nMC4_vtrain1.00-1.0 > delme_plotting 2>&1 
aprun -n 1 -N 1 python WeightAnalyser.py TrainingSim_Cluster_asymmetricTauij__50x2x1_0-400_taui200_nHC20_nMC4_vtrain1.00-1.0 > delme_plotting 2>&1 


#aprun -n 1 -N 1 python /cfs/milner/scratch/b/bkaplan/bcpnn-mt/PlottingScripts/PlotPrediction.py $1 > delme_plotting 2>&1
