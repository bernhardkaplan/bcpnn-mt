import utils
import simulation_parameters
import numpy as np
import CreateConnections as CC
#from mpi4py import MPI
#comm = MPI.COMM_WORLD

def prepare_sim(comm):

    if (comm != None):
        pc_id, n_proc = comm.rank, comm.size
    else:
        pc_id, n_proc = 0, 1

    # load simulation parameters
    network_params = simulation_parameters.parameter_storage()  # network_params class containing the simulation parameters
    params = network_params.load_params()                       # params stores cell numbers, etc as a dictionary
    tuning_prop = utils.set_tuning_prop(params['n_exc'], mode='hexgrid')        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
#    tuning_prop = utils.set_tuning_prop(params['n_exc'], mode='random')        # set the tuning properties of exc cells: space (x, y) and velocity (u, v)
    np.savetxt(params['tuning_prop_means_fn'], tuning_prop)
    x0, y0, u0, v0 = params['motion_params']
    motion = (x0, y0, u0, v0)
    #if (n_proc > 1): # distribute the number of minicolumns among processors
    my_units = utils.distribute_n(params['n_exc'], n_proc, pc_id)
    #else:
    #    my_units = (0, params['n_mc'])

    # create the input
    input_spike_trains = utils.create_spike_trains_for_motion(tuning_prop, motion, params, my_units) # write to paths defined in the params dictionary

    # create initial connections 
    # with weights based on cell's tuning properties
    weight_matrix, latency_matrix = CC.compute_weights_from_tuning_prop(tuning_prop, motion)

    # by random: # ugly function signature
#    CC.create_conn_list_by_random(params['conn_list_ee_fn_base']+'0.dat', (0, params['n_exc']), (0, params['n_exc']), params['p_ee'], params['w_ee_mean'], params['w_ee_sigma'])

    
    

    # initial connectivity is written to a file
    #CC.create_initial_connection_matrix(params['n_mc'], output_fn=params['conn_mat_init'], sparseness=params['conn_mat_init_sparseness'])

    # write inital bias values to file
    np.savetxt(params['bias_values_fn_base']+'0.dat', np.zeros(params['n_exc']))

    #input_vecs = utils.get_input_for_constant_velocity(tuning_prop, motion)  # compute f_in(t) for all cells; f_in(t) = time dependent input rate 
    #input_spike_trains = utils.convert_motion_energy_to_spike_trains(input_vecs)



