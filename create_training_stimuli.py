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

def create_training_stim_in_tp_space():
    print 'Using standard params'
    import simulation_parameters
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
    answer = raw_input('Overwrite training stimuli parameters file?\n\t%s\n' % (params['training_stimuli_fn']))
    if answer.capitalize() == 'Y':
        print 'Saving training stimuli parameters to:', params['training_stimuli_fn']
        np.savetxt(params['training_stimuli_fn'], training_stimuli)

    training_stim_duration = np.zeros(training_stimuli[:, 0].size)
    for i_ in xrange(training_stimuli[:, 0].size):
        stim_params = training_stimuli[i_, :]
        t_exit = utils.compute_stim_time(stim_params)
        print 'stim_params', stim_params, 't_exit', t_exit
        training_stim_duration[i_] = max(params['t_training_max'], t_exit)


    answer = raw_input('Overwrite training stimuli duration file?\n\t%s\n' % (params['training_stim_durations_fn']))
    if answer.capitalize() == 'Y':
        print 'Saving training stim durations to:', params['training_stim_durations_fn']
        np.savetxt(params['training_stim_durations_fn'], training_stim_duration)
    Plotter = Plotter(params)#, it_max=1)
    Plotter.plot_training_sample_space(plot_process=False)


def create_training_stimuli_based_on_tuning_prop(params, tp=None):

    if tp == None:
        tp = np.loadtxt(params['tuning_prop_exc_fn'])
    x_ = np.unique(np.around(tp[:, 0], decimals=1))
    v_ = np.unique(np.around(tp[:, 2], decimals=1))
    x_ = x_[np.where(x_ <= params['x_max_training'])[0]]
    x_ = x_[np.where(x_ >= params['x_min_training'])[0]]
    v_ = v_[np.where(v_ != 0.)[0]]
    N = x_.size * v_.size
    np.random.seed(params['visual_stim_seed'])
    mp = np.zeros((params['n_stim'], 4))
    cnt_ = 0
    while cnt_ != params['n_stim']:
        for i_x in xrange(x_.size):
            for i_v in xrange(v_.size):
                x_noise = 2 * params['training_stim_noise_x'] * np.random.random_sample() - params['training_stim_noise_x']
                v_noise = 2 * params['training_stim_noise_v'] * np.random.random_sample() + (1. + params['training_stim_noise_v'])
                mp[cnt_, 0] = x_[i_x] + x_noise
                mp[cnt_, 1] = .5
                mp[cnt_, 2] = v_[i_v] * v_noise
                cnt_ += 1
                if cnt_ == params['n_stim']:
                    break
            if cnt_ == params['n_stim']:
                break
                print 'noise', x_noise, v_noise
    idx = range(params['n_stim'])
    np.random.shuffle(idx)
    mp = mp[idx, :]
#    print 'Saving traingin stimuli to:', params['training_stimuli_fn']
    np.savetxt(params['training_stimuli_fn'], mp)
    training_stim_duration = np.zeros(params['n_stim'])
    for i_ in xrange(params['n_stim']):
        t_exit = utils.compute_stim_time(mp[i_, :])
        training_stim_duration[i_] = min(t_exit, params['t_training_max']) + params['t_stim_pause']
#    print 'Saving training stim durations to:', params['training_stim_durations_fn']
    np.savetxt(params['training_stim_durations_fn'], training_stim_duration)

    return mp[idx, :]



if __name__ == '__main__':

#    if len(sys.argv) == 2:
#        params = utils.load_params(sys.argv[1])
    print 'Using standard params'
    import simulation_parameters
    GP = simulation_parameters.parameter_storage()
    params = GP.params
    GP.write_parameters_to_file(params['params_fn_json'], params) # write_parameters_to_file MUST be called before every simulation
    tp, rfs = set_tuning_properties.set_tuning_properties_and_rfs_const_fovea(params)
    np.savetxt(params['tuning_prop_exc_fn'], tp)
    np.savetxt(params['receptive_fields_exc_fn'], rfs)
    create_training_stimuli_based_on_tuning_prop(params)
#    else:
#        create_training_stim_in_tp_space()

    pylab.show()
