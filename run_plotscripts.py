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

show = False

script = sys.argv[1]
if script == 'current' or script == 'currents':
    script_name = 'PlottingScripts/PlotCurrents.py'
    folders = sys.argv[2:]
elif script == 'prediction':
    script_name = 'PlottingScripts/PlotPrediction.py'
    folders = sys.argv[2:]
elif script == 'anticipation':
    script_name = 'PlottingScripts/PlotAnticipation.py'
    folders = sys.argv[2:]
else:
    folders = sys.argv[2:]
list_of_jobs = []

pure_names = ''
for f in folders:
    params = utils.load_params(f)
    n_stim = params['n_stim']
    stim_range = params['stim_range']
    for i_ in xrange(stim_range[0], stim_range[1]):
        if script == 'anticipation':
            cmd = 'python %s %s' % (script_name, f)
        else:
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
    n_my_jobs = len(range(my_idx[0], my_idx[1]))
    print 'pc_id %d job indices:' % pc_id, my_idx
    for j_, i_ in enumerate(range(my_idx[0], my_idx[1])):
#        print 'debug list_of_jobs:', list_of_jobs, 'i_', i_
        job_name = list_of_jobs[i_]
        print 'pc_id %d runs job nr %d / %d' % (pc_id, j_ + 1, n_my_jobs), job_name
        os.system(job_name)
else:
    print 'No MPI found'


if USE_MPI:
    comm.barrier()

if pc_id == 0 and show:
    if script == 'currents' or script == 'current':
        display_cmd = 'ristretto $(find %s -name prediction_stim*.png)' % pure_names
    display_cmd = 'ristretto $(find %s -name prediction_stim*.png)' % pure_names
    print 'display_cmd:', display_cmd
    os.system(display_cmd)
