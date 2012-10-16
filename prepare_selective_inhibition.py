import numpy as np
import utils

import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.load_params()                       # params stores cell numbers, etc as a dictionary

tp = np.loadtxt(params['tuning_prop_means_fn'])

# spatial distribution of inhibitory cells
hexgrid_pos = utils.set_hexgrid_positions(params, params['N_RF_X_INH'], params['N_RF_Y_INH'])
output_fn = params['inh_cell_pos_fn']
print 'debug hexgrid pos shape', hexgrid_pos.shape
print 'printing inh cell positions to:', output_fn
np.savetxt(output_fn, hexgrid_pos)

# for each exc cell calculate the predicted position of the dot from the cell's tuning prop
predicted_positions = utils.get_predicted_stim_pos(tp) # x_predicted = x + v [a.u.]
print "Predicted positions: ", params['predicted_positions_fn']
np.savetxt(params['predicted_positions_fn'], predicted_positions)

# calculate the ring of excitatory source cells for each target inhibitory cell
src_pos = predicted_positions
tgt_pos = hexgrid_pos
import CreateConnections as CC
n_ei_per_cell = int(round(params['p_ei'] * params['n_exc']))
print 'Each inh cell receives input from %d exc cells (p_ei = %.3f)' % (n_ei_per_cell, params['p_ei'])
output_indices, output_distances = CC.get_exc_inh_connections(src_pos, tgt_pos, tp, n=n_ei_per_cell)
output_fn1 = params['exc_inh_adjacency_list_fn']
output_fn2 = params['exc_inh_distances_fn']
print 'saving exc-inh connections', output_fn1
print 'saving exc-inh distances ', output_fn2
np.savetxt(output_fn1, output_indices)
np.savetxt(output_fn2, output_distances)
# TODO: convert 
#self.params['exc_inh_weights_fn'] = '%sexc_to_inh_indices.dat' % (self.params['connections_folder']) # same format as exc_inh_distances_fn, containing the exc - inh weights

