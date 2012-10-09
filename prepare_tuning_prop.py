import numpy as np
import utils
import sys

import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.load_params()                       # params stores cell numbers, etc as a dictionary


print 'n_cells=%d\tn_exc=%d\tn_inh=%d' % (params['n_cells'], params['n_exc'], params['n_inh'])
#params['blur_X'], params['blur_V'] = float(sys.argv[1]), float(sys.argv[2])
print 'Blur', params['blur_X'], params['blur_V']

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

tuning_prop = utils.set_tuning_prop(params, mode='hexgrid', v_max=params['v_max'])        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
print "Saving tuning_prop to file:", params['tuning_prop_means_fn']
np.savetxt(params['tuning_prop_means_fn'], tuning_prop)

# for each cells calculate the predicted position of the dot from the cell's tuning prop
predicted_positions = utils.get_predicted_stim_pos(tuning_prop) # x_predicted = x + v [a.u.]
print "Predicted positions: ", params['predicted_positions_fn']
np.savetxt(params['predicted_positions_fn'], predicted_positions)

# spatial distribution of inhibitory cells
hexgrid_pos = utils.set_hexgrid_positions(params, params['n_inh'])
output_fn = params['inh_cell_pos_fn']
print 'printing inh cell positions to:', output_fn
np.savetxt(output_fn, hexgrid_pos)

# calculate the ring of excitatory source cells for each target inhibitory cell
src_pos = predicted_positions
tgt_pos = hexgrid_pos
import CreateConnections as CC
output_indices, output_distances = CC.get_indices_in_vicinity(src_pos, tgt_pos, radius=.2, n=50)
output_fn1 = params['exc_inh_adjacency_list_fn']
output_fn2 = params['exc_inh_distances_fn']
print 'saving exc-inh connections', output_fn1
print 'saving exc-inh distances ', output_fn2
np.savetxt(output_fn1, output_indices)
np.savetxt(output_fn2, output_distances)
# TODO: convert 
#self.params['exc_inh_weights_fn'] = '%sexc_to_inh_indices.dat' % (self.params['connections_folder']) # same format as exc_inh_distances_fn, containing the exc - inh weights


mp = params['motion_params']
indices, distances = utils.sort_gids_by_distance_to_stimulus(tuning_prop, mp) # cells in indices should have the highest response to the stimulus
n = params['n_gids_to_record']
np.savetxt(params['gids_to_record_fn'], indices[:n], fmt='%d')
print 'Saving gids to record to: ', params['gids_to_record_fn']
