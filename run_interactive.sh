echo "Starting at `date`"
export CRAY_ROOTFS=DSL

. /opt/modules/default/etc/modules.sh
module swap PrgEnv-pgi PrgEnv-gnu
module add nest
#module add 10kproject
module add site-python

export PYTHONPATH=/pdc/vol/nest/2.2.2/lib64/python2.6/site-packages:/pdc/vol/python/packages/site-python-2.6/lib64/python2.6/site-packages:/cfs/klemming/nobackup/b/bkaplan/PythonPackages/lib64/python2.6/site-packages/

aprun -n 24 python /cfs/nobackup/b/bkaplan/bcpnn-mt/main_training.py

echo "Stopping at `date`"

