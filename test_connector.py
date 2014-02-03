
import numpy as np
import pyNN.hardware.brainscales as sim
import pyNN.space as space


import simulation_parameters
ps = simulation_parameters.parameter_storage()#fn)
params = ps.params

exc_pop = sim.Population(params['n_exc'], sim.IF_cond_exp, params['cell_params_exc'], label='exc_cells')
inh_pop = sim.Population(params['n_inh'], sim.IF_cond_exp, params['cell_params_inh'], label="inh_pop")
