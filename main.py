import utils
import os
import simulation_parameters
import numpy

# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
tuning_prop = utils.set_tuning_prop(params['n_mc'], mode='hexgrid')        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
numpy.savetxt(params['tuning_prop_means_fn'], tuning_prop)
x0, y0, u0, v0 = params['motion_params']
motion = (x0, y0, u0, v0)

# create the input
input_spike_trains = utils.create_spike_trains_for_motion(tuning_prop, motion, params) # write to paths defined in the params dictionary

#input_vecs = utils.get_input_for_constant_velocity(tuning_prop, motion)  # compute f_in(t) for all cells; f_in(t) = time dependent input rate 
#input_spike_trains = utils.convert_motion_energy_to_spike_trains(input_vecs)

# run the simulation
#os.system("python network_sim.py brian")

# analyze the output and run offline learning


