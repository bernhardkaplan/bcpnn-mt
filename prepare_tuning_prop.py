import numpy as np
import utils
import sys

import simulation_parameters
PS = simulation_parameters.parameter_storage()
params = PS.load_params()                       # params stores cell numbers, etc as a dictionary


print 'n_cells=%d\tn_exc=%d\tn_inh=%d' % (params['n_cells'], params['n_exc'], params['n_inh'])
PS.create_folders()
PS.write_parameters_to_file()

if params['n_grid_dimensions'] == 1:
    tuning_prop_exc = utils.set_tuning_prop_1D(params, cell_type='exc')
    tuning_prop_inh = utils.set_tuning_prop_1D(params, cell_type='inh')
else:
    tuning_prop_exc = utils.set_tuning_prop(params, mode='hexgrid', cell_type='exc')
    tuning_prop_inh = utils.set_tuning_prop(params, mode='hexgrid', cell_type='inh')

print "Saving exc tuning_prop to file:", params['tuning_prop_means_fn']
np.savetxt(params['tuning_prop_means_fn'], tuning_prop_exc)
print "Saving inh tuning_prop to file:", params['tuning_prop_inh_fn']
np.savetxt(params['tuning_prop_inh_fn'], tuning_prop_inh)

#print 'Calculating gids to record...'
#mp = params['motion_params']
#indices, distances = utils.sort_gids_by_distance_to_stimulus(tuning_prop_exc, mp, params) # cells in indices should have the highest response to the stimulus
#n = params['n_gids_to_record']
#np.savetxt(params['gids_to_record_fn'], indices[:n], fmt='%d')
#print 'Saving gids to record to: ', params['gids_to_record_fn']
