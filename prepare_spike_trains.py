import numpy as np
import utils
import sys

import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.load_params()                       # params stores cell numbers, etc as a dictionary
# either:
#params['blur_X'], params['blur_V'] = float(sys.argv[1]), float(sys.argv[2])
# or:
try: # try to get motion params from command line
    params['motion_params'] = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    seed = int(sys.argv[5])
except: # take the standard ones
    mp = params['motion_params']
    seed = params['input_spikes_seed']


PS.set_filenames()
PS.create_folders()
PS.write_parameters_to_file()

print 'n_cells=%d\tn_exc=%d\tn_inh=%d' % (params['n_cells'], params['n_exc'], params['n_inh'])
print 'Blur', params['blur_X'], params['blur_V']

scale_input_frequency = False
if scale_input_frequency:
    scaling_factor = utils.scale_input_frequency(params['blur_X'])
    params['f_max_stim'] *= scaling_factor

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

try:
    tuning_prop = np.loadtxt(params['tuning_prop_means_fn'])
except:
    print 'File with tuning properties missing: %s\nPlease run: \nmpirun -np [N] python prepare_tuning_prop.py\nOR\npython prepare_tuning_prop.py' % params['tuning_prop_means_fn']
    exit(1)

my_units = utils.distribute_n(params['n_exc'], n_proc, pc_id)
utils.create_spike_trains_for_motion(tuning_prop, params, contrast=.9, my_units=my_units, seed=seed) # write to paths defined in the params dictionary
if comm != None:
    comm.barrier()
