import numpy as np
import os
import SimulationManager
import time

# analysis modules
import plot_prediction

sigma_x_start = 0.6
sigma_x_step = 0.1
sigma_x_stop = sigma_x_start + 2 * sigma_x_step
sigma_x_range = np.arange(sigma_x_start, sigma_x_stop, sigma_x_step)

sigma_v_start = 0.1
sigma_v_step = 0.1
sigma_v_stop = sigma_v_start + 8 * sigma_v_step
sigma_v_range = np.arange(sigma_v_start, sigma_v_stop, sigma_v_step)
import simulation_parameters

PS = simulation_parameters.parameter_storage()

t_start = time.time()
simStarter = SimulationManager.SimulationManager(PS)

i_ = 0
w_input_exc = 2e-3
for sigma_v in sigma_v_range:
    for sigma_x in sigma_x_range:
        # -----   pre-computed connectivity 
#        new_params = {  'initial_connectivity' : 'precomputed', 'w_sigma_x' : sigma_x, 'w_sigma_v' : sigma_v}
        new_params = {  'initial_connectivity' : 'precomputed', 'w_sigma_x' : sigma_x, 'w_sigma_v' : sigma_v, 'w_input_exc' : w_input_exc}
        simStarter.update_values(new_params)

        # analysis 1
        plot_prediction.plot_prediction(simStarter.params)

        # copy files from the previous folder needed for the next simulation

#        new_params = { 'initial_connectivity' : 'random'}
        new_params = {  'initial_connectivity' : 'random',  'w_input_exc' : w_input_exc}
        simStarter.update_values(new_params)
        plot_prediction.plot_prediction(simStarter.params)
        i_ += 1

t_stop = time.time()
t_run = t_stop - t_start
print "Full analysis duration: %d sec or %.1f min for %d cells (%d exc, %d inh)" % (t_run, (t_run)/60., \
        simStarter.params['n_cells'], simStarter.params['n_exc'], simStarter.params['n_inh'])
