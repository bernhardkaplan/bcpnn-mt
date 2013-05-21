import numpy as np
import utils
import sys

import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.load_params()                       # params stores cell numbers, etc as a dictionary


print 'n_cells=%d\tn_exc=%d\tn_inh=%d' % (params['n_cells'], params['n_exc'], params['n_inh'])
#params['blur_X'], params['blur_V'] = float(sys.argv[1]), float(sys.argv[2])
#print 'Blur', params['blur_X'], params['blur_V']

PS.create_folders()
PS.write_parameters_to_file()

# not yet required 
#try:
#    from mpi4py import MPI
#    USE_MPI = True
#    comm = MPI.COMM_WORLD
#    pc_id, n_proc = comm.rank, comm.size
#    print "USE_MPI:", USE_MPI, 'pc_id, n_proc:', pc_id, n_proc
#except:
#    USE_MPI = False
#    pc_id, n_proc, comm = 0, 1, None
#    print "MPI not used"

tuning_prop_exc = utils.set_tuning_prop(params, mode='hexgrid', cell_type='exc')        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
print "Saving tuning_prop to file:", params['tuning_prop_means_fn']
np.savetxt(params['tuning_prop_means_fn'], tuning_prop_exc)


print 'Calculating gids to record...'
mp = params['motion_params']
indices, distances = utils.sort_gids_by_distance_to_stimulus(tuning_prop_exc, mp, params) # cells in indices should have the highest response to the stimulus
n = params['n_gids_to_record']
np.savetxt(params['gids_to_record_fn'], indices[:n], fmt='%d')
print 'Saving gids to record to: ', params['gids_to_record_fn']

tuning_prop_inh= utils.set_tuning_prop(params, mode='hexgrid', cell_type='inh')        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
print "Saving tuning_prop to file:", params['tuning_prop_inh_fn']
np.savetxt(params['tuning_prop_inh_fn'], tuning_prop_inh)
