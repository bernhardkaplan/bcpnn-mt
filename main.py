import utils


# load simulation parameters
network_params = utils.network_parameters()                 # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
tuning_prop = utils.set_cell_tunings(params['n_exc'])       # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
input_vecs = utils.get_input(tuning_prop, params['t_sim'])  # compute f_in(t) for all cells; f_in(t) = time dependent input rate 
input_spike_trains = utils.convert_motion_energy_to_spike_trains(input_vecs)

