import os
import sys
import numpy as np
import utils
#from PlottingScripts.PlotPrediction import plot_prediction

try: 
    from mpi4py import MPI
    USE_MPI = True
    comm = MPI.COMM_WORLD
    pc_id, n_proc = comm.rank, comm.size
    print "USE_MPI:", USE_MPI, 'pc_id, n_proc:', pc_id, n_proc
except:
    USE_MPI = False
    pc_id, n_proc, comm = 0, 1, None
    print "MPI not used"


script_name = 'PlottingScripts/PlotPrediction.py'

list_of_jobs = []

folders = sys.argv[1:]
pure_names = ''
for f in folders:
    params = utils.load_params(f)
    n_stim = params['n_stim']
    stim_range = params['stim_range']
    for i_ in xrange(stim_range[0], stim_range[1]):
        cmd = 'python %s %s %d %d' % (script_name, f, i_, i_+1)
        print 'cmd:', cmd
        if USE_MPI:
            list_of_jobs.append(cmd)
        else:
            os.system(cmd)
    pure_names += '%s ' % f


if USE_MPI:
    # distribute the commands among processes
    my_idx = utils.distribute_n(len(list_of_jobs), n_proc, pc_id) # this holds the indices for the jobs to be run by this processor
    print 'pc_id %d job indices:' % pc_id, my_idx
    for i_ in xrange(my_idx[0], my_idx[1]):
        job_name = list_of_jobs[i_]
        print 'pc_id %d runs job nr %d / %d' % (pc_id, i_, my_idx[1] - my_idx[0]), job_name
        os.system(job_name)
else:
    print 'No MPI found'


if pc_id == 0:
    display_cmd = 'ristretto $(find %s -name prediction_stim0.png)' % pure_names
    os.system(display_cmd)
