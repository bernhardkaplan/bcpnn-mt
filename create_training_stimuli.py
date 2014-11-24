import sys
import os
import CreateInput
import json
import simulation_parameters
import numpy as np
import time
import os
import utils
import pylab
from PlottingScripts.plot_training_samples import Plotter
import random
import set_tuning_properties

if __name__ == '__main__':


    GP = simulation_parameters.parameter_storage()
    params = GP.params
    GP.write_parameters_to_file(params['params_fn_json'], params) # write_parameters_to_file MUST be called before every simulation
    print 'n_cycles', params['n_training_cycles']
    np.random.seed(params['visual_stim_seed'])
    CI = CreateInput.CreateInput(params)

    tp, rfs = set_tuning_properties.set_tuning_properties_and_rfs_const_fovea(params)
    np.savetxt(params['tuning_prop_exc_fn'], tp)


    training_stimuli_sample = CI.create_training_sequence_iteratively()     # motion params drawn from the cells' tuning properties
    training_stimuli_grid = CI.create_training_sequence_from_a_grid()       # sampled from a grid layed over the tuning property space
    training_stimuli_center = CI.create_training_sequence_around_center()   # sample more from the center in order to reduce risk of overtraining action 0 and v_x_max
    training_stimuli = np.zeros((params['n_stim_training'], 4))
    n_grid = int(np.round(params['n_stim_training'] * params['frac_training_samples_from_grid']))
    n_center = int(np.round(params['n_stim_training'] * params['frac_training_samples_center']))
    random.seed(params['visual_stim_seed'])
    np.random.seed(params['visual_stim_seed'])
    training_stimuli[:n_center, :] = training_stimuli_center
    training_stimuli[n_center:n_center+n_grid, :] = training_stimuli_grid[random.sample(range(params['n_stim_training']), n_grid), :]

#    training_stimuli[:n_grid, :] = training_stimuli_grid[random.sample(range(params['n_stim_training']), n_grid), :]
#    training_stimuli[n_grid:n_grid+n_center, :] = training_stimuli_center 
    training_stimuli[n_grid+n_center:, :] = training_stimuli_sample[random.sample(range(params['n_stim_training']), params['n_stim_training'] - n_grid - n_center), :]
    print 'Saving training stimuli parameters to:', params['training_stimuli_fn']
    np.savetxt(params['training_stimuli_fn'], training_stimuli)

    training_stim_duration = np.zeros(training_stimuli[:, 0].size)
    for i_ in xrange(training_stimuli[:, 0].size):
        stim_params = training_stimuli[i_, :]
        t_exit = CI.compute_stim_time(stim_params)
        print 'stim_params', stim_params, 't_exit', t_exit
        training_stim_duration[i_] = max(params['t_training_max'], t_exit)


    Plotter = Plotter(params)#, it_max=1)
    Plotter.plot_training_sample_space(plot_process=True)
    print 'Saving training stim durations to:', params['training_stim_durations_fn']
    np.savetxt(params['training_stim_durations_fn'], training_stim_duration)
    pylab.show()
