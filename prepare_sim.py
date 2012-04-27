import utils
import simulation_parameters
import numpy as np
import CreateConnections as CC
from mpi4py import MPI
comm = MPI.COMM_WORLD
pc_id, n_proc = comm.rank, comm.size

# load simulation parameters
network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
tuning_prop = utils.set_tuning_prop(params['n_mc'], mode='random')        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
np.savetxt(params['tuning_prop_means_fn'], tuning_prop)
x0, y0, u0, v0 = params['motion_params']
motion = (x0, y0, u0, v0)


if (n_proc > 1): # distribute the number of minicolumns among processors
    my_units = utils.distribute_n(params['n_mc'], n_proc, pc_id)
else:
    my_units = (0, params['n_mc'])

# create the input
input_spike_trains = utils.create_spike_trains_for_motion(tuning_prop, motion, params, my_units) # write to paths defined in the params dictionary

# initial connectivity is written to a file
CC.create_initial_connection_matrix(params['n_mc'], output_fn=params['conn_mat_init'], sparseness=params['conn_mat_init_sparseness'])
exit(1)

#input_vecs = utils.get_input_for_constant_velocity(tuning_prop, motion)  # compute f_in(t) for all cells; f_in(t) = time dependent input rate 
#input_spike_trains = utils.convert_motion_energy_to_spike_trains(input_vecs)

# run the simulation
#os.system("python network_sim.py nest")

# analyze the output and run offline learning


